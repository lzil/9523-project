import argparse
import os
import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from nn_util import conv_layer, conv_to_fc_layer, fc_layer, maxout

PWD = os.path.dirname(os.path.realpath(__file__)) + '/'
TB_LOGS_DIR = PWD + 'tb_logs/'
MNIST_DIR = PWD + 'data/'
SIZE = 28
NUM_LABELS = 10
NUM_CHANNELS = 1

BATCH_SIZE = 100
NUM_EPOCHS = 100
EVAL_FREQUENCY = 10

timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")

def model(data, dropout=None, maxout_k=1, activation_fn=tf.nn.relu):
    data = tf.reshape(data, [BATCH_SIZE, SIZE, SIZE, NUM_CHANNELS])
    conv = conv_layer(data, depth=64, window=5,
            pool=(2, 2), maxout_k=maxout_k, activation_fn=activation_fn, name='conv1')
    conv = conv_layer(conv, depth=32, window=5,
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
    tensorboard_prefix = TB_LOGS_DIR + timestamp
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
    merged = tf.merge_all_summaries()
    metric_summaries = tf.merge_summary([loss_summary, accuracy_summary]) # skip weights

    start_time = time.time()
    with tf.Session() as sess:
        # Initialize summary writer for TensorBoard
        os.makedirs(tensorboard_prefix + 'training/')
        train_writer = tf.train.SummaryWriter(tensorboard_prefix + 'training/', graph=sess.graph)
        os.makedirs(tensorboard_prefix + 'validation/')
        val_writer = tf.train.SummaryWriter(tensorboard_prefix + 'validation/')

        # Run all the initializers to prepare the trainable parameters.
        sess.run(tf.initialize_all_variables())
        print('Initialized!')

        # Loop through training steps.
        for step in range(1, NUM_EPOCHS * mnist.train.num_examples // BATCH_SIZE + 1):

            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            data, labels = mnist.train.next_batch(BATCH_SIZE)

            # Run the optimizer to update weights can calculate loss
            _, train_l, train_predictions, train_accuracy, train_summary = sess.run(
                    [train_step, loss, prediction, accuracy, merged],
                    feed_dict={x: data, y: labels, keep_prob: 0.5})
            train_writer.add_summary(train_summary, step)

            # print some extra information once reach the evaluation frequency
            if step % EVAL_FREQUENCY == 0:
                # get next validation batch
                val_data, val_labels = mnist.validation.next_batch(BATCH_SIZE)

                # calculate validation set metrics
                val_l, val_predictions, val_accuracy, val_summary = sess.run(
                        [loss, prediction, accuracy, metric_summaries],
                        feed_dict={x: val_data, y: val_labels, keep_prob: 1.})
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
                print('\tValidation top-1 accuracy: %.1f%%' % (100*val_accuracy))
                sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dropout', default=False, action='store_true', help='Use Dropout.')
    parser.add_argument('-m', '--maxout', type=int, default=None, help='Use Maxout. Pass in number of activation inputs.')
    args = parser.parse_args()
    run(args)
