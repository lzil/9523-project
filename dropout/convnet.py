import argparse
import os
import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from nn_util import conv_layer, conv_to_fc_layer, fc_layer, maxout, weight_to_image_summary

PWD = os.path.dirname(os.path.realpath(__file__)) + '/'
TB_LOGS_DIR = PWD + 'tb_logs/'
MNIST_DIR = PWD + '../data/'
SIZE = 28
NUM_LABELS = 10
NUM_CHANNELS = 1

BATCH_SIZE = 100
NUM_EPOCHS = 10
EVAL_FREQUENCY = 10
TEST_FREQUENCY = 500
HISTOGRAM_FREQ = 100

timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")

def model(data, dropout=None, maxout_k=1, activation_fn=tf.nn.relu):
    data = tf.reshape(data, [BATCH_SIZE, SIZE, SIZE, NUM_CHANNELS])
    conv = conv_layer(data, depth=64, window=5,
            dropout=dropout,
            pool=(2, 2), maxout_k=maxout_k, activation_fn=activation_fn, name='conv1')
    conv = conv_layer(conv, depth=32, window=5,
            dropout=dropout,
            pool=(2, 2), maxout_k=maxout_k, activation_fn=activation_fn, name='conv2')
    reshape = conv_to_fc_layer(conv)
    hidden = fc_layer(
        reshape,
        depth=128,
        maxout_k=maxout_k,
        activation_fn=activation_fn,
        dropout=dropout,
        name='fc1'
    )
    output = fc_layer(
        hidden,
        depth=NUM_LABELS,
        maxout_k=1,
        activation=False,
        name='fc2'
    )
    return output

def run(args):
    if args.name:
        args.name += '__'
    tensorboard_prefix = TB_LOGS_DIR + args.name + timestamp + '/'
    if not os.path.exists(MNIST_DIR):
        os.makedirs(MNIST_DIR)
    mnist = input_data.read_data_sets(MNIST_DIR, one_hot=True)

    keep_prob = tf.placeholder(tf.float32, name='keep_prob') # for dropout
    if args.maxout is None:
        maxout_k = 1
        activation_fn = tf.nn.relu
    else:
        maxout_k = args.maxout
        activation_fn = maxout(args.maxout)
    dropout_train = args.dropout
    dropout_val = args.dropout_val
    dropout_test = args.dropout_val

    x = tf.placeholder(tf.float32, [BATCH_SIZE, SIZE*SIZE], name='input')
    y = tf.placeholder(tf.int32, [BATCH_SIZE, NUM_LABELS], name='labels')

    logits = model(x, dropout=keep_prob, maxout_k=maxout_k, activation_fn=activation_fn)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(loss)
    prediction = tf.nn.softmax(logits)
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    loss_summary = tf.scalar_summary('Loss', loss)
    accuracy_summary = tf.scalar_summary('Accuracy', accuracy)
    error_summary = tf.scalar_summary('Error', 1 - accuracy)
    merged = tf.merge_all_summaries()
    metric_summaries = tf.merge_summary([loss_summary, accuracy_summary, error_summary]) # skip weights

    start_time = time.time()
    with tf.Session() as sess:
        # Initialize summary writer for TensorBoard
        os.makedirs(tensorboard_prefix + 'training/')
        train_writer = tf.train.SummaryWriter(tensorboard_prefix + 'training/', graph=sess.graph)
        os.makedirs(tensorboard_prefix + 'validation/')
        val_writer = tf.train.SummaryWriter(tensorboard_prefix + 'validation/')
        os.makedirs(tensorboard_prefix + 'test/')
        test_writer = tf.train.SummaryWriter(tensorboard_prefix + 'test/')

        # Run all the initializers to prepare the trainable parameters.
        sess.run(tf.initialize_all_variables())
        print('Initialized!')

        def make_image(step):
            summary = weight_to_image_summary(tf.get_collection('conv_w')[0], name='weights/%d'%step)
            _summary = sess.run(summary)
            train_writer.add_summary(_summary)
            train_writer.flush()
            print('Added image summary.')

        def test(step):
            _acc = 0.
            for s in range(mnist.test.num_examples // BATCH_SIZE):
                test_data, test_labels = mnist.test.next_batch(BATCH_SIZE)

                # calculate testidation set metrics
                test_l, test_predictions, test_accuracy, test_summary = sess.run(
                        [loss, prediction, accuracy, metric_summaries],
                        feed_dict={x: test_data, y: test_labels, keep_prob: dropout_test})
                _acc += test_accuracy
            _acc /= mnist.test.num_examples // BATCH_SIZE

            # Add TensorBoard summary to summary writer
            test_writer.add_summary(sess.run(tf.scalar_summary('Test Accuracy', _acc)), step)
            test_writer.flush()

        make_image(0)

        # Loop through training steps.
        for step in range(1, NUM_EPOCHS * mnist.train.num_examples // BATCH_SIZE + 1):

            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            data, labels = mnist.train.next_batch(BATCH_SIZE)

            # Run the optimizer to update weights can calculate loss
            summary = merged if step % HISTOGRAM_FREQ == 0 else metric_summaries
            _, train_l, train_predictions, train_accuracy, train_summary = sess.run(
                    [train_step, loss, prediction, accuracy, summary],
                    feed_dict={x: data, y: labels, keep_prob: dropout_train})
            train_writer.add_summary(train_summary, step)


            # print some extra information once reach the evaluation frequency
            if step % EVAL_FREQUENCY == 0:
                # get next validation batch
                val_data, val_labels = mnist.validation.next_batch(BATCH_SIZE)

                # calculate validation set metrics
                val_l, val_predictions, val_accuracy, val_summary = sess.run(
                        [loss, prediction, accuracy, metric_summaries],
                        feed_dict={x: val_data, y: val_labels, keep_prob: dropout_val})
                val_writer.add_summary(val_summary, step)

                # Add TensorBoard summary to summary writer
                val_writer.add_summary(val_summary, step)
                train_writer.flush()
                val_writer.flush()

                # Print info/stats
                elapsed_time = time.time() - start_time
                start_time = time.time()

                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * BATCH_SIZE / mnist.train.num_examples,
                       1000 * elapsed_time / EVAL_FREQUENCY))
                print('\tMinibatch loss: %.3f' % train_l)
                print('\tMinibatch accuracy: %.1f%%' % (100*train_accuracy))
                print('\tValidation loss: %.3f' % val_l)
                print('\tValidation accuracy: %.1f%%' % (100*val_accuracy))
                sys.stdout.flush()

            if step % TEST_FREQUENCY == 0:
                test(step)

            if not step & (step - 1): # if power of 2
                make_image(step)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dropout', type=float, default=1., help='Use Dropout on training.')
    parser.add_argument('-f', '--dropout_val', type=float, default=1., help='Use Dropout on validation.')
    parser.add_argument('-m', '--maxout', type=int, default=None, help='Use Maxout. Pass in number of activation inputs.')
    parser.add_argument('-o', '--name', type=str, default='', help='A name for the run.')
    args = parser.parse_args()
    run(args)
