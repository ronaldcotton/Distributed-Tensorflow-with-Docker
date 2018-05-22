# /usr/bin/env python3
# -*- coding: utf-8 -*-

# pip3 install --upgrade tflearn tensorflow psutil scipy h5py

""" Deep Neural Network for MNIST dataset classification task.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

"""
# 12/27/17 - added model.json import
# 1/3/17 - made changes so that loss can be changed to cross_entropy
# 1/3/17 - made saving system based on hashing model layers to prevent
#        - crashes when loading models with different structures
# 1/3/17 - added names to layers for tensorboard legibility
# 1/4/17 - using argparse to load different json models
#        - if no argument is given, program loads model.json
# 1/5/17 - changed format of json file
# 1/8/17 - added method to have program run on single cpu, add to argument

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
import json  # for load and 
import sys  # for exit
import os.path  # if file exists
import hashlib
import argparse
import psutil
import multiprocessing
import random
import time
import csv
import functools
import operator
from base64 import b64encode
from tflearn.data_utils import shuffle, to_categorical


"""
md5B64Filename function
    creates unique filename depending on layers-models
    converts md5 to base64 generating a 24 character filename
"""
def md5B64Filename(s):  
    hash = hashlib.md5(s.encode('utf-8'))
    result = hash.digest()
    return b64encode(result).strip().decode('cp437')


"""
JSONObject
    using the object_hook with json.loads, creates a class given JSONdata
    with any key,value pair:
        any value with quotes is unicode
        any value without quotes will attempt to create the right type
"""
class JSONObject:
    def __init__(self, dict):
            vars(self).update(dict)
            
"""
appendCSV
    appends data to CSV file - incase of multiple runs
"""
def appendCSV(filename, line):
    with open(filename + ".csv", "a") as csv_file:
        writer = csv.writer(csv_file,  dialect="excel")
        writer.writerow(line)
"""
source: https://stackoverflow.com/questions/36668542/flatten-batch-in-tensorflow
https://stackoverflow.com/questions/43284684/reshape-batch-of-tensors-into-batch-of-vectors-in-tensorflow
"""
def shapeToOneD(input):
    n = functools.reduce(operator.mul, input.shape.as_list()[1:], 1)
    return tf.reshape(x, [-1, n])
    # tensor = tf.convert_to_tensor(input, dtype=tf.float32)
    # shape = tensor.get_shape().as_list()
    # dim = numpy.prod(shape[1:])
    # return tf.reshape(x, [-1, dim])
    
parse = argparse.ArgumentParser(description='DNN for MNIST Dataset')
parse.add_argument('-m','-model',dest='model', nargs='?', const='model.json', help='JSON model file', default='model.json')
parse.add_argument('-c','-cpu',dest='cpu', nargs='?', const=0, help='Processor to execute on', default=0)
args = parse.parse_args()

p = psutil.Process()
maxcpu = multiprocessing.cpu_count()-1
cpu = []
if maxcpu < int(args.cpu):
    cpu.append(random.randint(0,maxcpu))
else:
    cpu.append(int(args.cpu))
p.cpu_affinity(cpu)


if os.path.isfile(args.model):
    with open(args.model, 'r') as myfile:
        data = myfile.read().replace('\n','')  # turns file into single string
    jdata = json.loads(data, object_hook=JSONObject)
else:
    print ("ERROR: model.json file does not exist. Aborting.")
    sys.exit()


inputlayer = int(jdata.input_nodes)
outputlayer = int(jdata.output_nodes)
innerLayers = len(jdata.layers)-1
regressionLoss = str(jdata.layers[innerLayers].loss)

# reshape data to 1D tensor



# save layers for cvs model data
layers = []
layers.append(inputlayer)
for i in range(innerLayers):
    layers.append(jdata.layers[i].nodes)
layers.append(outputlayer)

if jdata.model == "mnist":  # input 784 - output 10
    print("http://yann.lecun.com/exdb/mnist/")
    import tflearn.datasets.mnist as mnist
    X, Y, testX, testY = mnist.load_data(one_hot=True)
elif jdata.model == "cifar10":  # input 1024 - output 10
    print("https://www.cs.toronto.edu/~kriz/cifar.html")
    from tflearn.datasets import cifar10
    (X, Y), (testX, testY) = cifar10.load_data()
    X, Y = shuffle(X, Y)
    Y = to_categorical(Y)
    testY = to_categorical(testY)
    X = shapeToOneD(X)
    Y = shapeToOneD(Y)
    testX = shapeToOneD(testX)
    testY = shapeToOneD(testY)
elif jdata.model == "cifar100":  # input 1024 - output 100
    print("https://www.cs.toronto.edu/~kriz/cifar.html")
    from tflearn.datasets import cifar100
    (X, Y), (testX, testY) = cifar100.load_data()
elif jdata.model == "oxflower17.py": # input 50176 - output 17
    print("http://www.robots.ox.ac.uk/~vgg/data/flowers/17/")
    from tflearn.datasets import oxflower17
    (X, Y) = oxflower17.load_data()
elif jdata.model == "svhn":  # input 1024 - output 10
    print("http://ufldl.stanford.edu/housenumbers")
    from tflearn.datasets import svhn
    X, Y, testX, testY = svhn.load_data()
else:
    sys.exit(1)

# Building deep neural network
net = tflearn.input_data(shape=[None, inputlayer], name='input')

modelFilename = ""
for i in range(innerLayers):
    modelFilename += str(jdata.layers[i].nodes) + str(jdata.layers[i].activation)[:5]
    net = tflearn.fully_connected(net, int(jdata.layers[i].nodes), bias=jdata.layers[i].bias, activation=str(jdata.layers[i].activation), name='dense'+str(i))
# output layer
net = tflearn.fully_connected(net, outputlayer, activation=str(jdata.layers[innerLayers].activation), name='output')

# Regression using SGD with learning rate
sgd = tflearn.SGD(learning_rate=jdata.learning_rate, lr_decay=1)
net = tflearn.regression(net, optimizer=sgd, metric=tflearn.metrics.Accuracy(),
                         loss=regressionLoss, batch_size=jdata.batch_size)

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)

modelFilename += str(innerLayers) + str(inputlayer)  + str(outputlayer) + regressionLoss[:2]
if jdata.load_model:
    if os.path.isfile( md5B64Filename(modelFilename) + ".tflearn.meta"):
        model.load( md5B64Filename(modelFilename) + ".tflearn" )
    else:
        print ("no previous model to load.")

# start timer
start = time.time()

model.fit(X, Y, n_epoch=jdata.epochs, validation_set=(testX, testY),
          batch_size=jdata.batch_size, show_metric=True, snapshot_epoch=True, run_id="dense_model")

# end timer
elapsedInSec = time.time()-start

trainacc = model.evaluate(X, Y)[0]
validationacc = model.evaluate(testX, testY)[0]
aveacc = (model.evaluate(X, Y)[0] + model.evaluate(testX, testY)[0])/2

if jdata.save_model:
    model.save(md5B64Filename(modelFilename) + ".tflearn" )
    
print(jdata.end_print)

CSVfile = args.model + '.csv'
csvData = []
csvData.append(aveacc)
csvData.append(trainacc)
csvData.append(validationacc)
csvData.append(jdata.model)
csvData.append(layers)
csvData.append(jdata.learning_rate)
csvData.append(jdata.batch_size)
csvData.append(elapsedInSec)
csvData.append(elapsedInSec/jdata.epochs)
csvData.append(jdata.load_model)
csvData.append(jdata.save_model)
csvData.append(md5B64Filename(modelFilename) + ".tflearn")
appendCSV(args.model, csvData)


