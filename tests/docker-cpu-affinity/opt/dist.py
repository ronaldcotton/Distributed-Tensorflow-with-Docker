# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Distributed MNIST training and validation, with model replicas.
A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on one parameter server (ps), while the ops
are executed on two worker nodes by default. The TF sessions also run on the
worker node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.
The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time
import datetime

import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
import csv

filetimestamp = datetime.now()

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/opt/MNIST-data",
                    "Directory for storing mnist data")
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 1,
                    "Total number of gpus for each machine."
                    "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("capture_steps", 10,
                     "capture data every x steps")
flags.DEFINE_integer("train_steps", 2000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
flags.DEFINE_string("ps_hosts","localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None,"job name: worker or ps")
flags.DEFINE_string("num_cpus", "0", "Comma-separated list of cpus")
FLAGS = flags.FLAGS

IMAGE_PIXELS = 28
SEED = 66478

def main(unused_argv):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, seed=SEED)

  if FLAGS.download_only:
    sys.exit(0)

  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index =="":
    raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)

  if FLAGS.job_name == 'worker':
    filename = startingCSVFile()

  # convert nums_cpus flag to list of ints
  cpuList = list(map(int, FLAGS.num_cpus.split(",")))

  # set cpu affinity using list
  setAffinity(cpuList)

  # Construct the cluster and start the server
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")

  # Get the number of workers.
  num_workers = len(worker_spec)

  cluster = tf.train.ClusterSpec({ "ps": ps_spec, "worker": worker_spec})

  if not FLAGS.existing_servers:
    # Not using existing servers. Create an in-process server.
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
      server.join()

  is_chief = (FLAGS.task_index == 0)

  if FLAGS.num_gpus > 0:
    # Avoid gpu allocation conflict: now allocate task_num -> #gpu
    # for each worker in the corresponding machine
    gpu = (FLAGS.task_index % FLAGS.num_gpus)
    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
  elif FLAGS.num_gpus == 0:
    # Just allocate the CPU to worker server
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)

  tf.set_random_seed(SEED)

  with tf.device(
      tf.train.replica_device_setter(
          worker_device=worker_device,
          ps_device="/job:ps/cpu:0",
          cluster=cluster)):

    global_step = tf.Variable(0, name="global_step", trainable=False)

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

    # Ops: located on the worker specified with FLAGS.task_index
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, [None, 10])

    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)

    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))

    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

    if FLAGS.sync_replicas:
      if FLAGS.replicas_to_aggregate is None:
        replicas_to_aggregate = num_workers
      else:
        replicas_to_aggregate = FLAGS.replicas_to_aggregate

      opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=replicas_to_aggregate,
          total_num_replicas=num_workers,
          name="mnist_sync_replicas")

    train_step = opt.minimize(cross_entropy, global_step=global_step)

    if FLAGS.sync_replicas:
      local_init_op = opt.local_step_init_op
      if is_chief:
        local_init_op = opt.chief_init_op

      ready_for_local_init_op = opt.ready_for_local_init_op

      # Initial token and chief queue runners required by the sync_replicas mode
      chief_queue_runner = opt.get_chief_queue_runner()
      sync_init_op = opt.get_init_tokens_op()

    init_op = tf.global_variables_initializer()
    train_dir = tempfile.mkdtemp()

    if FLAGS.sync_replicas:
      sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir=train_dir,
          init_op=init_op,
          local_init_op=local_init_op,
          ready_for_local_init_op=ready_for_local_init_op,
          recovery_wait_secs=1,
          global_step=global_step)
    else:
      sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir=train_dir,
          init_op=init_op,
          recovery_wait_secs=1,
          global_step=global_step)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

    print(sess_config)

    # The chief worker (task_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
      print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
      print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)

    if FLAGS.existing_servers:
      server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
      print("Using existing server at: %s" % server_grpc_url)

      sess = sv.prepare_or_wait_for_session(server_grpc_url,
                                            config=sess_config)
    else:
      sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    print("Worker %d: Session initialization complete." % FLAGS.task_index)

    if FLAGS.sync_replicas and is_chief:
      # Chief worker will start the chief queue runner and call the init op.
      sess.run(sync_init_op)
      sv.start_queue_runners(sess, [chief_queue_runner])

    # Perform training
    time_begin = time.time()
    ms = str(time_begin - int(time_begin))[1:7]
    print("Training begins @ %s%s" % (str(datetime.fromtimestamp(int(time_begin)).strftime('%Y-%m-%d %H:%M:%S')), ms))

    local_step = 0
    losses = []
    avg_loss_sz = FLAGS.capture_steps
    lasttime = 0

    while True:
      # Training feed
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
      train_feed = {x: batch_xs, y_: batch_ys}

      _, step, loss_val  = sess.run([train_step, global_step, cross_entropy], feed_dict=train_feed)
      local_step += 1

      now = time.time()

      if len(losses) == avg_loss_sz:
        losses.pop(0)
      losses.append(loss_val)
      
      if local_step % FLAGS.capture_steps == 0:
        timestamp = str(datetime.fromtimestamp(int(now)).strftime('%Y-%m-%d %H:%M:%S'))
        ms = str(now - int(now))[1:7]
        loss = float(np.mean(losses))
        acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
        print('%s%s : worker/%d global: %d step: %d loss: %f accuracy: %f' % (timestamp, ms, FLAGS.task_index, step, local_step, loss, acc))
        if lasttime == 0:
          writeCSVFile(filename, timestamp, ms, FLAGS.task_index, step, local_step, loss, acc, 0)
        else:
          writeCSVFile(filename, timestamp, ms, FLAGS.task_index, step, local_step, loss, acc, now - lasttime)
        lasttime = now

        if ( FLAGS.train_steps < step ):
          break

    sv.stop()
    time_end = time.time()
    ms = str(time_end - int(time_end))[1:7]
    print("Training ends @ %s%s" % (str(datetime.fromtimestamp(int(time_end)).strftime('%Y-%m-%d %H:%M:%S')), ms))
    training_time = time_end - time_begin
    print("[TRAINING] elapsed time: %f secs - task_id %d" % (training_time, FLAGS.task_index))

    # Validation feed
    val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    val_xent = sess.run(cross_entropy, feed_dict=val_feed)
    print("After %d training step(s), validation cross entropy = %g, loss_val = %f" % (FLAGS.train_steps, val_xent, np.mean(losses)))

    print('test accuracy %g' % sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))


def hexDateTime():
  """ create a unique hex timestamp for file generation
      down to the millisecond
  """
  return str(format(int(filetimestamp.strftime("%y%j%H%M%S%f")), 'x'))


def tensorToNPArray(Tensor):
  """ convert tensor -> NPArray """
  return tf.Session().run(Tensor)


def tensorToElement(Tensor, index):
  """ convert tensor -> Element """
  return tf.Session().run(Tensor)[index]


def startingCSVFile():
  filename = str(FLAGS.task_index) + '-' + hexDateTime() + '.csv'

  with open(filename, 'w') as csvfile:
    csv.writer(csvfile).writerow(['time', 'worker_task_index', 'global_step', 'local_step', 'loss', 'acc', 'time delta'])
  return filename


def writeCSVFile(filename, timestamp, ms, task_index, step, local_step, loss, acc, delta):
  time = timestamp + ms
  with open(filename, 'a') as csvfile:
    csv.writer(csvfile).writerow([time, task_index, step, local_step, loss, acc, delta])


def setAffinity(CPUArray):
  """ setAffinity()
      defined to work for Windows, Linux, FreeBSD
      does not work in MACOSX
      undefined behavior for RiscOS(*), Cygwin(*), AtheOS, OS/2

      Input: list of cpus to devote to process, zero-indexed
      Output: process will only run on that enviornment
      Results:

          if running on MacOSX, AtheOS or OS/2 exit

          if list contains numbers outside of the range of cpus,
          below 0 and above maxcpu these values will be removed from the list.
          if the list is empty, then the affinity of this program is set
          to all available processors.

          set affinity will then be set to all processors available.

  """
  from platform import platform
  from multiprocessing import cpu_count
  import psutil
  pf = platform(terse=1).lower()[:6]
  if pf == 'darwin' or pf == 'os2emx' or pf== 'atheos':
    return  # cases in which cpu_affinity doesn't work
  maxcpu = cpu_count()-1
  for cpu in CPUArray:
    if cpu > maxcpu or cpu < 0:
      CPUArray.remove(cpu)  # remove CPUS that don't exist from list

  if not CPUArray:  # if the array is empty, use all available processors
    CPUA = list(range(cpu_count()))
  else:  # otherwise, use the processors requested
    CPUA = list(CPUArray)
  psutil.Process().cpu_affinity(CPUA)


if __name__ == "__main__":
  tf.app.run()
