# -*- coding: utf-8 -*-

# this ONLY optimizes the number of layers
# (considering a nn with no dropout)
# use iterable to create all versions of layers
# generate dnn-json.py compatibile json file (w/ filename based on number)
# make bash script that loads evenly over all processors

import json
import hashlib
from os import chmod
from multiprocessing import cpu_count
import argparse
import itertools
import math
import random

"""
    writeJson
    output Json to file
"""

def writeJson(filename, data):
    with open(filename, 'w') as outfile:  
        json.dump(data, outfile)

"""
    str2bool
    source: https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
"""
def str2bool(v):
  return v.lower() in ("true", "1")

"""
 nCr - n(Choose)R = n!/(r!(n-r)!)  { where n >= r >= 0 & both n & r int }
 Binomial Coefficient - Combinations without Repetition
"""

def nCr(n, r):
    try:
        return int(math.factorial(n)/(math.factorial(r)*math.factorial(n-r)))
    except Exception as e:
        print ("ERROR: ", e)
        return None

"""
 nPr - n(Permutations)r = n!/(n-r)!  { where n >= r >= 0 & both n & r int }
 Premutations without Repetition
"""

def nPr(n, r):
    try:
        return int(math.factorial(n)/math.factorial(n-r))
    except Exception as e:
        print ("ERROR: ", e)
        return None
        
"""
 cartesianSquare
 Premutations with repetition
"""
def cartesianSquare(n, r):
    try:
        return int(math.pow(n,r))
    except Exception as e:
        print ("ERROR: ", e)
        return None

if __name__ == '__main__':
    # nodemax depends on number of layers - built for minst
    lastLayer = inputLayer = 784
    outputLayer = 10
    processors = cpu_count()-1
    cpu = 0

    # if inputLayer is 5674, for example, nodemax rounds up to 6000
    nodemax = int(math.ceil((int(str(inputLayer)[0])+1)*10**(len(str(inputLayer))-1)))

    parse = argparse.ArgumentParser(description='hyperparameter optimizer using grid method')
    parse.add_argument('-v','-version',dest='version', nargs='?', const='1', help='JSON model version (int)', default='1')
    parse.add_argument('-m','-model',dest='model', nargs='?', const='mnist', help='JSON model dataset (string)', default='mnist')
    parse.add_argument('-l','-load',dest='load', nargs='?', const='false', help='JSON model loading setting (bool)', default='false')
    parse.add_argument('-s','-save',dest='save', nargs='?', const='false', help='JSON model save boolean setting (bool)', default='false')
    parse.add_argument('-e','-epochs',dest='epochs', nargs='?', const='10', help='JSON model num of epochs (int)', default='10')
    parse.add_argument('-b','-batchsize',dest='batchsize', nargs='?', const='64', help='JSON model batch size (int)', default='64')
    parse.add_argument('-lr','-learningrate',dest='learningrate', nargs='?', const='0.01', help='JSON model learning rate (int)', default='0.01')
    # layers min and max
    parse.add_argument('-lmin','-layermin',dest='layermin', nargs='?', const='1', help='JSON model layers min (int)', default='1')
    parse.add_argument('-lmax','-layermax',dest='layermax', nargs='?', const='5', help='JSON model layers max (int)', default='5')
    # steps are calculated by number of iterations
    # parse.add_argument('-st','-step',dest='step', nargs='?', const='20', help='JSON model skip every {x} nodes (int)', default='20')
    # offset for steps
    parse.add_argument('-o','-offset',dest='offset', nargs='?', const='0', help='JSON model offset for steps', default='0')
    # nodes per layer - min and max
    parse.add_argument('-nmin','-nodemin',dest='nodemin', nargs='?', const='10', help='JSON model nodes min (int)', default='10')
    parse.add_argument('-nmax','-nodemax',dest='nodemax', nargs='?', const=nodemax, help='JSON model nodes max (int)', default=nodemax)
    parse.add_argument('-n', '-number', dest='number', nargs='?', const='10000', help='JSON model number (int)', default='10000')
    parse.add_argument('-i','-iter',dest='iter', nargs='?', const='combo', help='combo, prem, prod', default='combo')
    parse.add_argument('-r','-random',dest='random', action='store_true', help='make all values random')
    parse.set_defaults(random=False)
    args = parse.parse_args()

    # loop through layermin up to layermax
    listOfNodesInLayer = list(range(int(args.nodemin)+int(args.offset), int(args.nodemax)))   
    for r in range(int(args.layermin), int(args.layermax)+1):
        print "Processing Layer " + str(r) + " of " + str(int(args.layermax)+1)
        step = 1
        if args.iter == 'prod':
            while cartesianSquare(len(listOfNodesInLayer), r) > int(args.number):
                step += 1
                listOfNodesInLayer = list(range(int(args.nodemin)+int(args.offset), int(args.nodemax), step))
                listSize = len(listOfNodesInLayer)
                if args.random:
                    try:
                        listOfNodesInLayer = random.sample(range(int(args.nodemin)+int(args.offset), int(args.nodemax)), listSize)
                    except ValueError:  # list larger than samples
                        listOfNodesInLayer = random.sample(range(int(args.nodemin)+int(args.offset), int(args.nodemin)+int(args.offset)+listSize+1), listSize)
            possible = list(itertools.product(listOfNodesInLayer, repeat=r))

        elif args.iter == 'prem':
            while nPr(len(listOfNodesInLayer), r) > int(args.number):
                step += 1
                listOfNodesInLayer = list(range(int(args.nodemin)+int(args.offset), int(args.nodemax), step))
                listSize = len(listOfNodesInLayer)
                if args.random:
                    try:
                        listOfNodesInLayer = random.sample(range(int(args.nodemin)+int(args.offset), int(args.nodemax)), listSize)
                    except ValueError:  # list larger than samples - make range one bigger than number of elements
                        listOfNodesInLayer = random.sample(range(int(args.nodemin)+int(args.offset), int(args.nodemin)+int(args.offset)+listSize+1), listSize)
            possible = list(itertools.permutations(listOfNodesInLayer, r))

        else:  # combo
            while nCr(len(listOfNodesInLayer), r) > int(args.number):
                step += 1
                listOfNodesInLayer = list(range(int(args.nodemin)+int(args.offset), int(args.nodemax), step))
                listSize = len(listOfNodesInLayer)
                if args.random:
                    try:
                        listOfNodesInLayer = random.sample(range(int(args.nodemin)+int(args.offset), int(args.nodemax)), listSize)
                    except ValueError:  # list larger than samples
                        listOfNodesInLayer = random.sample(range(int(args.nodemin)+int(args.offset), int(args.nodemin)+int(args.offset)+listSize+1), listSize)
            possible = list(itertools.combinations(listOfNodesInLayer, r))

        # r is the number of layers in the current model
        print possible[0]
        if r == 1:
            print possible
        for index, plist in enumerate(possible):  # p - index of possible
            print "plist = ",plist
            # fill known json data
            jdata = {}
            jdata['version'] = args.version
            jdata['model'] = args.model
            # only works for mnist
            jdata['input_nodes'] = inputLayer
            jdata['output_nodes'] = outputLayer
            jdata['load_model'] = str2bool(args.load)
            jdata['save_model'] = str2bool(args.save)
            jdata['iterations'] = 1
            jdata['epochs'] = int(args.epochs)
            jdata['batch_size'] = int(args.batchsize)
            jdata['learning_rate'] = float(args.learningrate)
            # add json jdata for this layer
            jdata['layers'] = []
            print "r = ", r
            for q in range(r):  # q - index of layers in possible list
                jdata['layers'].append({"type": "dense", "activation": "tanh", "nodes": plist[q], "bias": True})
                # ~ s = '{"type": "dense", "nodes": "' + str(possible[p][q]) + '", "bias": true }'                    
                # ~ jdata['layers'].append(s)
            # add output layer
            jdata['layers'].append({"type":"output", "activation":"softmax", "loss": "softmax_categorical_crossentropy"})
            jdata['end_print'] = 'Layer #' + str(r) + ' of ' + str(args.layermax) + ' - Test #' + str(index) + ' of ' + str(len(possible)-1)
            writeJson(str(r)+'_'+str(index).zfill(len(args.number))+'.json', jdata)
            
            # add to shell script alternating processor
            with open("processor" + str(cpu) + ".sh", "a") as myfile:
                myfile.write('python dnn-json.py -model ' + str(r)+'_'+str(index).zfill(len(args.number))+'.json' + ' -c ' + str(cpu) + '\n')
                myfile.write('sleep 1\n')
            cpu += 1
            if cpu > processors:
                cpu = 0

    # turn on execute bit for programs
    for c in range(processors+1):
        chmod("processor" + str(c) + ".sh", 0755)  # 0o755 in py3
