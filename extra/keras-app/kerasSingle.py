#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# backwards compatibility to Python 2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import TensorFlow
import tensorflow as tf

# import kerasimport keras
import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# import h5py
import h5py

# import numpy
import numpy as np

# regular imports
import argparse
from psutil import cpu_count
from sys import exit, argv, stderr
import os
import urllib
import time
from datetime import datetime
import tfDataSet
import json
import hashlib
import collections
import csv


now = datetime.now()  # called only once per run

def hexDateTime():
    """ create a unique hex timestamp for file generation
        down to the millisecond
    """
    return str(format(int(now.strftime("%y%j%H%M%S%f")), 'x'))


def saveJson(filename, data):
  """ save JSON file -- compat in py 2 & 3 """
  try:
    with open(filename, 'w') as outfile:  # handles closing file
      json.dump(data, outfile)
  except Exception as e:
    eprint('JSON File ' + str(filename) + ' didn\'t save, aborting.')
    exit(1)


def loadJson(filename):
  """ load JSON file -- compat. in py 2 & 3 """
  try:
    with open(filename, 'r') as infile:  # handles closing file
      data = json.load(infile)
  except Exception as e:
    eprint('JSON File ' + str(filename) + ' didn\'t load, aborting.')
    exit(1)
  return data


def saveAny(filename, data):
  """ save yaml file -- compat in py 2 & 3 """
  try:
    with open(filename, 'w') as outfile:  # handles closing file
      outfile.write(str(data))
  except Exception as e:
    eprint('ERROR: File ' + str(filename) + ' didn\'t save, aborting.' )
    exit(1)


def loadAny(filename):
  """ save yaml file -- compat in py 2 & 3 """
  try:
    with open(filename, 'r') as infile:  # handles closing file
      data = infile.readlines()
  except Exception as e:
    eprint('ERROR: File ' + str(filename) + ' didn\'t save, aborting.' )
    exit(1)
  return data

def activateKerasBackend():
  """ ensures that keras backend is being used --
      other backends include Theanos & CNTK
  """
  os.environ['KERAS_BACKEND'] = 'tensorflow'


def eprint(*args, **kwargs):
  """ prints to stderr - requires '__future__ import print_function' """
  print(*args, file=stderr, **kwargs)


def computerInfo():
  """ prints basic info about computer running software """
  from socket import getfqdn, gethostname, gethostbyname
  from platform import python_implementation, python_build, platform, machine, processor, node
  from psutil import cpu_count
  fqdn = getfqdn()
  hostname = gethostname()
  print(python_implementation(), python_build()[0])
  print(platform())
  print(machine(), processor(), "x", cpu_count(logical=False), "cores | ", cpu_count(logical=True), "threads")
  print("FQDN:", node(), fqdn)
  print("LAN IPv4:",gethostbyname(hostname))


def setSeed(reproducable):
  """ use the same random seed to make reproducable results or not """
  if reproducable:
    seed = 1234
    np.random.seed(seed)
  else:
    timeStampSeed = int(time.mktime(datetime.now().timetuple()))
    np.random.seed(timeStampSeed)


def setMKLVariables():
  """ set KMP, OML and tensorflow settings - does nothing if not KML
      batch_size and model effects optimial KMP/OMP settings
  """
  os.environ["KMP_BLOCKTIME"] = str(FLAGS.kmp_blocktime)
  os.environ["KMP_SETTINGS"] = str(FLAGS.kmp_settings)
  os.environ["KMP_AFFINITY"] = str(FLAGS.kmp_affinity)
  os.environ["OMP_NUM_THREADS"] = str(FLAGS.threads)


def setDataFormat(x_train, x_test, colors):
  """ set data format - optimizes data for Archetecture
      For example, if you have the MNIST dataset
      this will convert 786 elements into ---> 28x28x1
  """
  rows = tfDataSet.ROWS
  cols = tfDataSet.COLS

  if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], colors, rows, cols)
    x_test = x_test.reshape(x_test.shape[0], colors, rows, cols)
    input_shape = (colors, rows, cols)
  else:
    x_train = x_train.reshape(x_train.shape[0], rows, cols, colors)
    x_test = x_test.reshape(x_test.shape[0], rows, cols, colors)
    input_shape = (rows, cols, colors)
  return (x_train, x_test), input_shape


def flattenToRowVectors(x_train, y_train):
    """ Converts image data to 1 Dimension """
    size = tfDataSet.ROWS * tfDataSet.COLS * tfDataSet.COLORS
    x_train = x_train.reshape(TRAINELEMENTS, size)
    x_test = x_test.reshape(TESTELEMENTS, size)
    return x_train, xtest


def convertVectorsToBinary(y_train, y_test):
  """ Converts a class vector (integers) to binary class matrix.
      for use with categorical_crossentropy.
  """
  y_train = keras.utils.to_categorical(y_train, tfDataSet.NUMCLASS)
  y_test = keras.utils.to_categorical(y_test, tfDataSet.NUMCLASS)
  return y_train, y_test


def importDataFile(importName):
  """ imports python module, checks that file is in current path
      returns module
  """
  if os.path.isfile(importName + '.py'):  # if directory exists
    return __import__(importName)  # returns module
  else:
    eprint('ERROR: importDataFile(): unable to import file = "' + importName + '" does not exist.')
    exit(1)


def noFunction():
    """ error code for callTensforflowFunction if function does not exist """
    eprint('ERROR: Function does not exist.')
    exit(3)

def startingCSVFile():
  filename = FLAGS.savedir + hexDateTime() + '.csv'

  with open(filename, 'w', newline='') as csvfile:
    csv.writer(csvfile).writerow(['epoch', 'batch', 'val_loss', 'val_acc', 'loss', 'acc', 'start', 'end', 'delta msec', 'is_epoch'])

def writeCSVFile(epoch, batch, valLoss, valAcc, trainLoss, trainAcc):
  """ writes batches and epochs to file """
  filename = FLAGS.savedir + hexDateTime() + '.csv'
  curr = datetime.now()

  if valLoss is not None:
    isEpoch = True
    delta = (curr - epochStart)
  else:
    isEpoch = False
    delta = (curr - batchStart)

  with open(filename, 'a', newline='') as csvfile:
    csv.writer(csvfile).writerow([epoch, batch, valLoss, valAcc, trainLoss, trainAcc, batchStart.strftime("%Y-%m-%d %H:%M:%S"), curr.strftime("%Y-%m-%d %H:%M:%S"), delta, isEpoch])

def setEpoch(epoch):
    global epochStart
    epochStart = datetime.now()
    global epochnum
    epochnum = epoch

def setBatch():
    global batchStart
    batchStart = datetime.now()


# TODO: add fit_generator see - https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
# add zca whitining, featurewise normalization
def callTensorflowFunction(function, args, model, datagen, fn, m):
  """ processes a single line of modelList and creates required callbacks """
  callDict = {'Conv2D': Conv2D, 'MaxPooling2D': MaxPooling2D, 'Dropout': Dropout, 'Flatten': Flatten, 'Dense': Dense }
  if function == 'Add':
    tfFunction = callDict.get(args['function'], noFunction)
    args.pop('function', None)
    return model.add(tfFunction(**args))
  if function == 'Compile':
    return model.compile(**args)
  if function[:3] == 'Fit':  # if either Fit or Fit_generator
    # create callbacks as needed for model
    csvFile = FLAGS.savedir + hexDateTime() + '.csv'
    tbFile = FLAGS.logdir + hexDateTime()
    if hasattr(m, 'modelStop'):  # if m.modelStop exists
      cb_EarlyStopping = keras.callbacks.EarlyStopping(**m.modelStop)
    cb_TensorBoard = keras.callbacks.TensorBoard(log_dir = tbFile, histogram_freq=1, write_graph=True, write_grads=True)
    cb_CSVLog = keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch,logs: setEpoch(epoch),
                                                on_epoch_end=lambda epoch,logs: writeCSVFile(epoch, None, logs['val_loss'], logs['val_acc'], logs['loss'], logs['acc']),
                                                on_batch_begin=lambda batch,logs: setBatch(),
                                                on_batch_end=lambda batch,logs: writeCSVFile(epochnum, batch, None, None, logs['loss'], logs['acc']),
                                                on_train_begin=lambda logs: startingCSVFile())


    # add callbacks to model when runnning [] = no callbacks,
    # [cb_TensorBoard] = one callbacks
    # [cb_TensorBoard, cb_CSVLogger] = 2 callbacks, etc.

    args['callbacks'] = []
    if hasattr(m, 'modelStop'):  # if m.modelStop exists
      eprint("Callback: EarlyStopping Enabled.")
      args['callbacks'].append(cb_EarlyStopping)
    else:
      eprint("EarlyStopping callback not in use.")
    if m.modelConfig['cb_TensorBoard']:
        eprint("Callback: Tensorboard Enabled")
        print('execute "tensorboard --logdir=' + str(FLAGS.logdir) + '" to visualize data.' )
        args['callbacks'].append(cb_TensorBoard)
    if m.modelConfig['cb_CSVLogger']:
        eprint("Callback: CVSLog")
        args['callbacks'].append(cb_CSVLog)
    model.summary()  # print out model when all model.add() have been complete
  if function == 'Fit':
    # start = datetime.now()
    ret = model.fit(**args)
    # end = datetime.now()
    # storeTimeStampCSV(start, end)
    return ret
  if function == 'Fit_generator':
    if datagen is not None:
      args['generator'] = fn
      # start = datetime.now()
      ret = model.fit_generator(**args)
      # end = datetime.now()
      # storeTimeStampCSV(start, end)
      return ret
  if function == 'datagen.flow':
      return datagen.flow(**args)

# unsure how to use history for Model.fit: see https://stackoverflow.com/questions/36952763/how-to-return-history-of-validation-loss-in-keras

def generateVarDict(m):
  """ allows elements to be accessible by model.py """
  varDict = {}
  varDict['INPUTSHAPE'] = input_shape
  varDict['UNITS'] = tfDataSet.NUMCLASS
  varDict['X'] = train_x
  varDict['Y'] = train_y
  varDict['BATCHSIZE'] = FLAGS.batch_size
  if 'epochs' in m.modelConfig:
    varDict['EPOCHS'] = m.modelConfig['epochs']
  else:
    varDict['EPOCHS'] = 1
  if 'verbose' in m.modelConfig:
    varDict['VERBOSE'] = m.modelConfig['verbose']
  else:
    varDict['VERBOSE'] = 2
  varDict['VALIDATIONDATA'] = (test_x, test_y)
  if 'steps_per_epochs' in m.modelConfig:
    varDict['STEPSPEREPOCHS'] = m.modelConfig['verbose']
  return varDict


def processModel(modelName):
  """ loads model as defined by model.py """
  call = None

  m = importDataFile(modelName)

  model = Sequential()

  # generate dictionary with key, value pairs with variable values == value
  varDict = generateVarDict(m)

  # if image, use datagen = ImageDataGenerator(...)
  if m.modelConfig['datatype'] == 'image':
    datagen = ImageDataGenerator(**m.imageDataGenerator)
    datagen.fit(train_x)
  else:
    datagen = None

  modelstr = ""
  for index, outerDict in enumerate(m.modelList, 0):
    modelstr += json.dumps(str(outerDict))  # used for saving
    for outerKey, innerDict in outerDict.items():
      for key, value in varDict.items():
        for k, v in innerDict.items():
          if str(key) == str(v):
            innerDict[k] = varDict[key]
    call = callTensorflowFunction(outerKey, innerDict, model, datagen, call, m)

  # when complete to a final evaluation
  score = model.evaluate(x=test_x, y=test_y, verbose=2)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  # convert modelstr to hash
  modelstr = modelstr.encode('utf-8')
  hash = hashlib.md5(modelstr).hexdigest()
  return model, hash

def saveWholeModel(model, saveFolder):
  """ saves archetecture, weights, and optimizer states into a unique WebNN
      file """
  hdf5File = saveFolder + 'WebNN-all-'+ hexDateTime() + '.h5'
  print("Saving " + hdf5File)
  model.save(hdf5File)  # model = keras.models.load_model(<filename>) to


def loadWholeModel(loadFolder, loadFile):
  """ loads architecture, weights, and optimizer states from WebNN file
      returns: keras model
  """
  return keras.models.load_model(loadFolder + loadFile)


def saveArchitecture(saveFolder, model):
  """ saves keras archetecture in json and yaml format """
  jsonFile = saveFolder + 'WebNN-arch-'+ hexDateTime() + '.json'
  saveJson(jsonFile, model.to_json())
  yamlFile = saveFolder + 'WebNN-arch-'+ hexDateTime() + '.yaml'
  saveAny(yamlFile, model.to_yaml())


def loadJSONArchitecture(loadFolder, loadFile):
  """ loads keras archetecture in json format
      returns: keras model
  """
  json = loadJson(loadFolder + loadFile)
  return keras.models.model_from_json(json)


def loadYAMLArchitecture(loadFolder, loadFile):
  yaml = loadAny(loadFolder + loadFile)
  return keras.models.model_from_yaml(yaml)


def saveWeights(saveFolder, hash):
  """ saves Weights, must have the same archetecture when loading but the
      number of nodes can change in layers, but the initial weights for those
      layers are undefined.
  """
  weightsFile = saveFolder + 'WebNN-weights-'+ str(hash) + '.h5'
  model.save_weights(weightsFile)

def loadWeights(loadFolder, loadFile):
  """ loads Weights, must have the same archetecture (with exception of nodes)
  """
  model.load_weights(loadFolder + loadFile, by_name=True)


def setLogLevel(logLevel):
  """ set loglevel
        loglevel = 0 (all logs)
                   1 (filter info)
                   2 (filter info, warning)
                   3 (filter info, warning, error)
                   4 (fatal errors only)
  """
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = logLevel


def setLogLevelVerbose(level):
  """ set loglevel verbosity for tensorflow """
  if level == 0:
    tf.logging.set_verbosity(tf.logging.DEBUG)
  elif level == 1:
    tf.logging.set_verbosity(tf.logging.INFO)
  elif level == 2:
    tf.logging.set_verbosity(tf.logging.WARN)
  elif level == 3:
    tf.logging.set_verbosity(tf.logging.ERROR)
  else:
    tf.logging.set_verbosity(tf.logging.FATAL)


def printVisualization(element, classification):
  """ visualizes data to confirm that data has been loaded
      uses two characters horizonatal for every one vertical image
      to keep ratio
  """
  rows = len(element)
  cols = len(element[0])
  min = np.amin(element)
  max = np.amax(element)
  ave = np.average(element)
  betweenminave = (min + ave) / 2
  betweenavemax = (ave + max) / 2
  print(" |"*cols)
  for i in range(rows):
    for j in range(cols):
      pixel = element[i][j]
      if min <= pixel <= betweenminave:
        print("..", end='', sep='')
      elif betweenminave < pixel <= ave:
        print("░░", end='', sep='')
      elif ave < pixel <= betweenavemax:
        print("▒▒", end='', sep='')
      elif betweenavemax < pixel:
        print("▓▓", end='', sep='')
    print("")
  print("Value of Element: " + str(classification))

def makeDirs(dirname):
  if not os.path.isdir(dirname):
    os.makedirs(dirname)

def parseCommandArgs():
  """Take arguments from the command line to modify aspects of runtime

  arguments have defaults

  Args:
  None

  Returns:
  (tuple) FLAGS, unparsed
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Initial learning rate.')
  parser.add_argument('--max_steps', type=int, default=2000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type=int, default=100,
                      help='Batch size.  Must divide evenly into dataset size.')
  parser.add_argument('--datadir', type=str,
                      default='/host/data/',
                      help='Directory to put the input data.')
  parser.add_argument('--logdir', type=str,
                      default='/host/logs/',
                      help='Directory to put the log data.')
  parser.add_argument('--savedir', type=str,
                      default='/host/save/',
                      help='Directory to put the save data.')
  parser.add_argument('--loglevel', type=str, default='0',
                      help='set enviromental vairable TF_CPP_MIN_LOG_LEVEL'),
  parser.add_argument('--loglevelverbose', type=int, default='3',
                     help='set verbosity for tf.logging.set_verbosity()')
  parser.add_argument('--kmp_blocktime', type=str, default='1',
                      help='KML: sets time in ms for thread waiting after execution of parrallel region (0-30 best)')
  parser.add_argument('--kmp_affinity', type=str,
                      default='granularity=fine,verbose,compact,1,0',
                      help='KML: Enables binding threads to physical processing units.')
  parser.add_argument('--kmp_settings', type=bool, default=False,
                      help='KML: prints OpenMP variables')
  # psutil.cpu_count(logical=False)=cores psutil.cpu_count(logical=True)=threads
  parser.add_argument('--threads', type=int,
                      default=cpu_count(logical=True),
                      help='MKL: sets optimial number of threads')
  parser.add_argument('--cores', type=int,
                      default=cpu_count(logical=False),
                      help='MKL: sets optimial number of cores')
  # new arguments, overrides JSON file if implemented
  parser.add_argument('--nodes', nargs='+', type=int,
                      help='nodes, starting with input and ending with output')
  parser.add_argument('--activation', nargs='+', type=str,
                      help='activation per layer')
  parser.add_argument('--bias', nargs='+', type=bool,
                      help='boolean, turn on or off bias per layer')
  parser.add_argument('--batch_normal', nargs='+', type=bool,
                      help='boolean, turn on or off activation per layer')
  parser.add_argument('--dropout', nargs='+', type=float,
                      help='set dropout for each layer. 0-100 per layer')
  parser.add_argument('--dataset', type=str, default='MNIST',
                      help='choose dataset to use')
  parser.add_argument('--model', type=str, default='model',
                      help='choose model to load')
  parser.add_argument('--reproducable', type=bool, default=False,
                      help='set random seed or not')
  return parser.parse_known_args()


if __name__ == '__main__':
  """ runs the entire model """
  FLAGS, unparsed = parseCommandArgs()
  makeDirs(FLAGS.savedir)
  makeDirs(FLAGS.logdir)
  makeDirs(FLAGS.datadir)
  setSeed(FLAGS.reproducable)
  computerInfo()
  activateKerasBackend()
  setLogLevel(FLAGS.loglevel)
  setLogLevelVerbose(FLAGS.loglevelverbose)
  setMKLVariables()
  global train_x, train_y, test_x, test_y, input_shape
  (train_x, train_y), (test_x, test_y) = tfDataSet.importDataSet(FLAGS.dataset, dir='/host')
  print("1st Training Element:")
  printVisualization(train_x[0], train_y[0])
  print("1st Testing Element:")
  printVisualization(test_x[0], test_y[0])
  (train_x, test_x), input_shape = setDataFormat(train_x, test_x, tfDataSet.COLORS)
  (train_y, test_y) = convertVectorsToBinary(train_y, test_y)
  model, hash = processModel(FLAGS.model)  # hash needed for some save types
  saveWholeModel(model, FLAGS.savedir)
  saveArchitecture(FLAGS.savedir, model)
