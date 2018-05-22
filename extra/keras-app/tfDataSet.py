# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import keras
from keras.datasets import mnist, cifar10, cifar100
from keras import backend as K
import time
from datetime import datetime
import os


# bolded vs. camelcase to represent Globals
ROWS = None  # number of rows in table or rows in image
COLS = None  # number of cols in table or cols in image
NUMINPUT = None  # number of inputs, nodes
NUMCLASS = None  # number of outputs, nodes
DATASET = None  # dataset handler (?)
TRAINELEMENTS = None  # number of elements for train
TESTELEMENTS = None  # number of elements for test
CSVFILENAMETRAIN = None  # set filename for CSV
CSVFILENAMETEST = None  # set filename for CSV
CSVCLASSIFER = None  # set classifier for CSV
CSVHEADER = None  # set CSV header
DATASETLIST = ["MNIST", "CIFAR10", "CIFAR100", "IRIS", "CSV"]  # all available datasets
COLORS = None

def setStats(ytest, xtrain, ytrain):
  """ sets trainingelements, testingelements, rows, cols, numinputs
      for 2D images
  """
  global TRAINELEMENTS, TESTELEMENTS, ROWS, COLS, NUMINPUT, NUMCLASS
  TRAINELEMENTS = len(ytrain)
  TESTELEMENTS = len(ytest)
  ROWS = len(xtrain[0])  # y-axis
  COLS = len(xtrain[0][0])  # x-axis
  NUMINPUT = ROWS * COLS  # number of inputs, nodes
  NUMCLASS = len(np.unique(ytrain))


def importDataSet(dataset, dir='/host'):  # FLAGS.input_data_dir
  """ gets the data of the dataset from internet, stores locally in dir
      if dataset was not set, or incorrect, does nothing.
      returns:
        (None, None), (None, None) if no dataset exists or is unimplemented
        (XTRAIN, YTRAIN), (XTEST, YTEST) if ran properly
  """
  global COLORS
  (XTRAIN, YTRAIN), (XTEST, YTEST) = (None, None), (None, None)
  cwd = os.getcwd()
  os.chdir(dir)  # get current directory - save in correct path
  if dataset not in DATASETLIST:
    print("Model not defined.")
  if dataset == DATASETLIST[0]:  # mnist
    (XTRAIN, YTRAIN), (XTEST, YTEST) = mnist.load_data()
    os.chdir(cwd)
    # convert data from (0 - max) --> (0 - 1) floats
    XTRAIN = XTRAIN.astype('float32')
    XTEST = XTEST.astype('float32')
    XTRAIN /= np.amax(XTRAIN)
    XTEST /= np.amax(XTRAIN)
    COLORS = 1
    setStats(YTEST, XTRAIN, YTRAIN)
  if dataset == DATASETLIST[1]: # cifar10
    (XTRAIN, YTRAIN), (XTEST, YTEST) = cifar10.load_data()
    os.chdir(cwd)
    # convert data from (0 - max) --> (0 - 1) floats
    XTRAIN = XTRAIN.astype('float32')
    XTEST = XTEST.astype('float32')
    XTRAIN /= np.amax(XTRAIN)
    XTEST /= np.amax(XTRAIN)
    COLORS = 3
    setStats(YTEST, XTRAIN, YTRAIN)
  if dataset == DATASETLIST[2]:  # cifar100
    (XTRAIN, YTRAIN), (XTEST, YTEST) = cifar100.load_data(label_mode='fine')
    os.chdir(cwd)
    XTRAIN = XTRAIN.astype('float32')
    XTEST = XTEST.astype('float32')
    XTRAIN /= np.amax(XTRAIN)
    XTEST /= np.amax(XTRAIN)
    COLORS = 3
    setStats(YTEST, XTRAIN, YTRAIN)
  if dataset == DATASETLIST[3]:  # iris - todo: modifyfor keras - CSV import
    pass
  if dataset == DATASETLIST[4]:  # csv
    print("CSV: not yet implemented.")
  print("XTrain shape: " + str(XTRAIN.shape))
  print("YTrain shape: " + str(YTRAIN.shape))
  return (XTRAIN, YTRAIN), (XTEST, YTEST)
