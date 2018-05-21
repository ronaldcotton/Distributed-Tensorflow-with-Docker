# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# adapted from original code at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import datetime
import time
import math

import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
import csv

filetimestamp = datetime.now()

FLAGS = None
IMAGE_PIXELS = 28
SEED = 66478

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # use fixed random seed to remove/reduce randomness between tests
  tf.set_random_seed(SEED)

  # Variables of the hidden layer
  hid_w = tf.Variable(
      tf.truncated_normal(
          [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
          stddev=1.0 / IMAGE_PIXELS),
      name="hid_w")
  hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

  # Variables of the softmax layer
  sm_w = tf.Variable(
      tf.truncated_normal(
          [FLAGS.hidden_units, 10],
          stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
      name="sm_w")
  sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

  # input and output
  x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
  y_ = tf.placeholder(tf.float32, [None, 10])

  hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
  hid = tf.nn.relu(hid_lin)

  y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))

  # calculate accuracy
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  # Define loss and optimizer
  cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
  opt = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  filename = startingCSVFile()

  # Perform training
  time_begin = time.time()
  ms = str(time_begin - int(time_begin))[1:7]
  print("Training begins @ %s%s" % (str(datetime.fromtimestamp(int(time_begin)).strftime('%Y-%m-%d %H:%M:%S')), ms))

  local_step = 0
  losses = []
  avg_loss_sz = FLAGS.capture_steps
  lasttime = 0

  # Train
  for _ in range(FLAGS.train_steps):
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
    _, loss_val = sess.run([opt, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    local_step += 1
    step = local_step

    now = time.time()

    if local_step % FLAGS.capture_steps == 0:
      timestamp = str(datetime.fromtimestamp(int(now)).strftime('%Y-%m-%d %H:%M:%S'))
      ms = str(now - int(now))[1:7]
      loss = float(np.mean(losses))
      acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
      print('%s%s : worker/%d global: %d step: %d loss: %f accuracy: %f' % (timestamp, ms, 0, step, local_step, loss_val, acc))
      if lasttime == 0:
        writeCSVFile(filename, timestamp, ms, 0, step, local_step, loss_val, acc, 0)
      else:
        writeCSVFile(filename, timestamp, ms, 0, step, local_step, loss_val, acc, now - lasttime)
      lasttime = now

  time_end = time.time()
  ms = str(time_end - int(time_end))[1:7]
  print("Training ends @ %s%s" % (str(datetime.fromtimestamp(int(time_end)).strftime('%Y-%m-%d %H:%M:%S')), ms))
  training_time = time_end - time_begin
  print("[TRAINING] elapsed time: %f secs" % (training_time))

def hexDateTime():
  """ create a unique hex timestamp for file generation
      down to the millisecond
  """
  return str(format(int(filetimestamp.strftime("%y%j%H%M%S%f")), 'x'))

def tensorToElement(Tensor, index):
  """ convert tensor -> Element """
  return tf.Session().run(Tensor)[index]

def startingCSVFile():
  filename = 'results-' + hexDateTime() + '.csv'

  with open(filename, 'w') as csvfile:
    csv.writer(csvfile).writerow(['time', 'worker_task_index', 'global_step', 'local_step', 'loss', 'acc', 'time delta'])
  return filename

def writeCSVFile(filename, timestamp, ms, task_index, step, local_step, loss, acc, delta):
  time = timestamp + ms
  with open(filename, 'a') as csvfile:
    csv.writer(csvfile).writerow([time, task_index, step, local_step, loss, acc, delta])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='./MNIST-data', help='Directory for storing mnist data')
  parser.add_argument('--hidden_units', type=int, default=100, help='number of nodes in hidden units')
  parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
  parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps to perform')
  parser.add_argument('--batch_size', type=int, default=100, help='Training Batch Size')
  parser.add_argument('--capture_steps', type=int, default=10, help='capture data every x steps')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
