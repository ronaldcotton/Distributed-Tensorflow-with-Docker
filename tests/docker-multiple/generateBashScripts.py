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

def main():
  # write the first script
  f = open(FLAGS.filename, "w")

  # output the bash tag at the start
  f.write("#!/bin/bash\n")

  # make ps lines
  for i in range(10, 10+FLAGS.ps):
    f.write("ps{}ip='{}.{}'\n".format(i-10, FLAGS.ip, i))

  # make worker lines
  for i in range(100, 100+FLAGS.workers):
    f.write("worker{}ip='{}.{}'\n".format(i-100, FLAGS.ip, i))

  # write ip commands
  for i in range(10,10+FLAGS.ps):
    f.write("ps{}ip_cmd='--net tfdocker --ip {}.{}'\n".format(i-10, FLAGS.ip, i))

  # write worker commands
  for i in range(100, 100+FLAGS.workers):
    f.write("worker{}ip_cmd='--net tfdocker --ip {}.{}'\n".format(i-100, FLAGS.ip, i))

  # limit resources
  f.write("limitresources='--cpus={} --memory={}M'\n".format(int(multiprocessing.cpu_count()/(FLAGS.ps + FLAGS.workers)), int(FLAGS.memory)))

  # start tag
  f.write("mnist_replica_cmd='python {} --batch_size {} --num_gpu {} --train_steps {} --ps_hosts=".format('/opt/' + FLAGS.dist_program, FLAGS.batch_size, FLAGS.gpu, FLAGS.train_steps))

  # add ps hosts
  for i in range(10,10+FLAGS.ps):
    f.write("{}.{}:{}".format(FLAGS.ip, i, FLAGS.port))
    if i != (10+FLAGS.ps-1):
      f.write(",")

  # start tag 2
  f.write(" --worker_hosts=")

  # add workers
  for i in range(100, 100+FLAGS.workers):
    f.write("{}.{}:{}".format(FLAGS.ip, i, FLAGS.port))
    if i != (100+FLAGS.workers-1):
      f.write(",")

  f.write("'\n")
  # create the workers
  f.write("{} network create --driver=bridge --subnet={}.0/24 --gateway={}.1 tfdocker\n{} network create --driver=bridge tfdocker\n".format(FLAGS.command, FLAGS.ip, FLAGS.ip, FLAGS.command))

  # build the docker
  f.write("{} build -t ubuntu/tensorflow .\n".format(FLAGS.command))

  for i in range(FLAGS.workers):
    f.write("{} run -t -d -v {}/opt:/opt $limitresources $worker{}ip_cmd --name tfworker{} ubuntu/tensorflow\n".format(FLAGS.command, os.getcwd(), i, i))

  for i in range(FLAGS.ps):
    f.write("{} run -t -d -v {}/opt:/opt $limitresources $ps{}ip_cmd --name tfps{} ubuntu/tensorflow\n".format(FLAGS.command, os.getcwd(), i, i))

  for i in range(FLAGS.ps):
    f.write("{} exec -d tfps{} $mnist_replica_cmd --job_name=ps --task_index={}\n".format(FLAGS.command, i, i))

  for i in range(1, FLAGS.workers):
    f.write("{} exec -d tfworker{} $mnist_replica_cmd --job_name=worker --task_index={}\n".format(FLAGS.command, i, i))

  f.write("{} exec -i tfworker0 $mnist_replica_cmd --job_name=worker --task_index=0\n".format(FLAGS.command))

  # close file
  f.close()
  os.chmod(FLAGS.filename, 0755)

  # write second script
  f = open(FLAGS.filename_end, "w")

  # output the bash tag at the start
  f.write("#!/bin/bash\n")

  for i in range(FLAGS.ps):
    f.write("{} stop tfps{}\n".format(FLAGS.command, i))

  for i in range(FLAGS.workers):
    f.write("{} stop tfworker{}\n".format(FLAGS.command, i))

  for i in range(FLAGS.ps):
    f.write("{} container rm tfps{}\n".format(FLAGS.command, i))

  for i in range(FLAGS.workers):
    f.write("{} container rm tfworker{}\n".format(FLAGS.command, i))

  f.write("{} network rm tfdocker\n".format(FLAGS.command))

  # process all csv files
  f.write('for f in opt/*.csv; do tail -n +2 "$f" >> opt/temp.csv; done\n')
  f.write('echo "$(head -n 1 opt/0*.csv)\n$(sort -t \',\' -k 3 -g opt/temp.csv)" > ps{}workers{}.csv\n'.format(FLAGS.ps, FLAGS.workers))
  f.write("rm -f opt/*.csv\n")
  
  # close file
  f.close()

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
  parser.add_argument('--train_steps', type=int, default=10000,
                      help='number of training steps')
  parser.add_argument('--mem_percent', type=float, default=0.90,
                      help='memory usage for each instance from free memory')
  parser.add_argument('--port', type=int, default=2222,
                      help='port for all data communications')
  parser.add_argument('--command', type=str, default='docker',
                      help='set as "docker" or "sudo docker"')
  parser.add_argument('--memory', type=int, default=4096,
                      help='the amount of memory per instance')
  return parser.parse_known_args()

if __name__ == "__main__":
  FLAGS, unparsed = parseCommandArgs()
  main()
