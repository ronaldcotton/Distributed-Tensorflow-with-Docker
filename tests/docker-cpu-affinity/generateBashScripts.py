#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# backwards compatibility to Python 2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing
import os
# import psutil - psutil.virtual_memory().free * mem_percent

def loopingList(iter, start, about):
  """ generates a list
      INPUT:
        iter - number of iterations to loop
        start - the starting value
        about - number to reach before starting back at zero
      OUTPUT:
        tuple: comma-delimited string, last element
  """
  res = []
  for i in range(iter):
    res.append((i + start) % about)
  return (str(res).strip('[]').replace(' ', ''), ((i + start + 1) % about)) # converts list to comma-delimited string

def main():
  # write the first script
  print("generating {}...".format(FLAGS.filename))
  f = open(FLAGS.filename, "w")
  cwd = os.getcwd()

  # output the bash tag at the start
  f.write("#!/bin/bash\n\n")

  # make directory for opt - and copy dist file to $PATH/opt
  f.write("mkdir {}/opt\n".format(cwd))
  f.write("cp {}/dist.py {}/opt\n\n".format(cwd, cwd))

  # make ip string & ip command
  f.write("ip='{}.10'\n".format(FLAGS.ip))
  f.write("ip_cmd='--net tfdocker --ip {}.10'\n".format(FLAGS.ip))

  # start tag
  f.write("mnist_replica_cmd='python {} --batch_size {} --num_gpu {} --train_steps {} --ps_hosts=".format('/opt/' + FLAGS.dist_program, FLAGS.batch_size, FLAGS.gpu, FLAGS.train_steps))

  # variables for startPSPort and startWorkerPort
  startPSPort = 64000
  startWorkerPort = 64000 + FLAGS.ps

  # add ps hosts
  for i in range(startPSPort,startPSPort+FLAGS.ps):
    f.write("{}.10:{}".format(FLAGS.ip, i))
    if i != (startPSPort + FLAGS.ps - 1):
      f.write(",")

  # start tag 2
  f.write(" --worker_hosts=")

  # add workers
  for i in range(startWorkerPort, startWorkerPort+FLAGS.workers):
    f.write("{}.10:{}".format(FLAGS.ip, i))
    if i != (startWorkerPort + FLAGS.workers - 1):
      f.write(",")

  f.write("'\n\n")

  # create the workers
  f.write("{} network create --driver=bridge --subnet={}.0/24 --gateway={}.1 tfdocker\n{} network create --driver=bridge tfdocker\n\n".format(FLAGS.command, FLAGS.ip, FLAGS.ip, FLAGS.command))

  # build the docker
  f.write("{} build -t ubuntu/tensorflow .\n\n".format(FLAGS.command))

  f.write("{} run -t -d -v {}/opt:/opt $ip_cmd --name tf ubuntu/tensorflow\n\n".format(FLAGS.command, os.getcwd()))

  # handles case where the number of cpus is a minimum of one
  max_cpus = multiprocessing.cpu_count()
  processes = FLAGS.ps + FLAGS.workers
  num_cpus = int(max_cpus/processes)
  if max_cpus < processes:
    remaining_cpus = 0
  else:
    remaining_cpus = int(max_cpus % processes)
  if num_cpus == 0:
      num_cpus = 1

  print("cpus detected: {}, processes to run: {}, number of cpus per process: {}...".format(max_cpus, processes, num_cpus))

  # downloads the MNIST dataset
  f.write("{} exec -i tf python {} --download_only\n".format(FLAGS.command, '/opt/' + FLAGS.dist_program))
  f.write("sleep {}\n".format(FLAGS.delay))

  j = 0
  for i in range(FLAGS.ps):
    (output_cpus, j) = loopingList(num_cpus, j, max_cpus)
    f.write("{} exec -d tf $mnist_replica_cmd --job_name=ps --task_index={} --num_cpus={}\n".format(FLAGS.command, i, output_cpus))

  for i in range(1, FLAGS.workers):
    (output_cpus, j) = loopingList(num_cpus, j, max_cpus)
    f.write("{} exec -d tf $mnist_replica_cmd --job_name=worker --task_index={} --num_cpus={}\n".format(FLAGS.command, i, output_cpus))

  f.write("sleep {}\n".format(FLAGS.delay))
  (output_cpus, j) = loopingList(num_cpus + remaining_cpus, j, max_cpus)  # add in the remainder of cpus to last task
  f.write("{} exec -i tf $mnist_replica_cmd --job_name=worker --task_index=0 --num_cpus={}\n".format(FLAGS.command, output_cpus))

  # close file
  print("saving {}...".format(FLAGS.filename))
  f.close()
  os.chmod(FLAGS.filename, 0755)

  # write second script
  print("generating {}...".format(FLAGS.filename_end))
  f = open(FLAGS.filename_end, "w")

  # output the bash tag at the start
  f.write("#!/bin/bash\n")

  f.write("{} stop tf\n\n".format(FLAGS.command))
  f.write("{} container rm tf\n\n".format(FLAGS.command))
  f.write("{} network rm tfdocker\n\n".format(FLAGS.command))

  # process all csv files
  f.write('for f in opt/*.csv; do tail -n +2 "$f" >> opt/temp.csv; done\n')
  f.write('echo "$(head -n 1 opt/0*.csv)\n$(sort -t \',\' -k 3 -g opt/temp.csv)" > ps{}workers{}.csv\n'.format(FLAGS.ps, FLAGS.workers))
  f.write("rm -f opt/*.csv\n")

  # close file
  print("saving {}...".format(FLAGS.filename_end))
  f.close()
  os.chmod(FLAGS.filename_end, 0755)


def parseCommandArgs():
  """Take arguments from the command line to modify aspects of runtime

  arguments have defaults

  Args:
  None

  Returns:
  (tuple) FLAGS, unparsed
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--filename', type=str, default='start.sh',
                      help='filename of generated start script')
  parser.add_argument('--filename_end', type=str, default='stop.sh',
                      help='filename of generated end script')
  parser.add_argument('--dist_program', type=str, default='dist.py',
                      help='python program that is distributed')
  parser.add_argument('--ip', type=str, default='10.20.30',
                      help='start of all the ips')
  parser.add_argument('--workers', type=int, default=6,
                      help='number of processes to run')
  parser.add_argument('--ps', type=int, default=1,
                      help='number of parameter servers')
  parser.add_argument('--gpu', type=int, default=0,
                      help='number of gpu')
  parser.add_argument('--batch_size', type=int, default=64,
                      help='size of batches')
  parser.add_argument('--train_steps', type=int, default=20000,
                      help='number of training steps')
  parser.add_argument('--mem_percent', type=float, default=0.90,
                      help='memory usage for each instance from free memory')
  parser.add_argument('--command', type=str, default='docker',
                      help='set as "docker" or "sudo docker"')
  parser.add_argument('--memory', type=int, default=4096,
                      help='the amount of memory per instance')
  parser.add_argument('--delay', type=str, default='3s',
                      help='delay after downloading MNIST and before starting chief')
  return parser.parse_known_args()

if __name__ == "__main__":
  FLAGS, unparsed = parseCommandArgs()
  main()
