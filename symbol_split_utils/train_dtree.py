#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 14:24:22 2016

@author: chernov
"""

from os import environ, path, system
import sys
import random

import cv2
import numpy as np

main_dir = path.dirname(path.dirname(__file__))
if not main_dir in sys.path:
    sys.path.append(main_dir)

from marking_tools.markers_db_tools import flush_db, flush_db_dupl

DATASET_NAME = "symbols"
DATASET_NEGATIVE_NAME = "symbols_negative"
DATA_PATH = environ["BEORGDATAGEN"] + "/CData_full"
DATASET_NEGATIVE_PATH = DATA_PATH + "/" + DATASET_NEGATIVE_NAME
DATASET_PATH = DATA_PATH + "/" + DATASET_NAME

CLASSIFIER_DIR = environ["BEORGDATAGEN"] + "/cv_trained_classifiers/"
CLASSIFIER_NM1_PATH = CLASSIFIER_DIR + "trained_classifierNM1.xml"
CLASSIFIER_NM2_PATH = CLASSIFIER_DIR + "trained_classifierNM2.xml"
CLASSIFIER_NM1_BIN = CLASSIFIER_DIR + "bin/extract_featuresNM1"
CLASSIFIER_NM2_BIN = CLASSIFIER_DIR + "bin/extract_featuresNM2"
NM1_CVS = CLASSIFIER_DIR + "char_datasetNM1.csv"
NM2_CVS = CLASSIFIER_DIR + "char_datasetNM2.csv"

random.seed()

def load_csv(filename):
    num_lines = sum(1 for line in open(filename))
    features = len(open(filename).readline().split(",")) - 1
    
    samples = np.zeros((num_lines, features), np.float32)
    responses = np.zeros(num_lines, np.float32)
    
    i = 0
    for line in open(filename):
        fields = line.replace("\n","").split(",")
        if fields[0] == "C": 
            responses[i] = 0.
        else:
            responses[i] = 1.
        for j in range(features):
           samples[i][j] = float(fields[j + 1])
        i += 1
    return samples, responses

        
def load_base(fn):
    a = np.loadtxt(fn, np.float32, delimiter=',', 
                   converters={ 0 : lambda ch : int(not(ord(ch) == ord("C")))})
    samples, responses = a[:,1:], a[:,0]
    return samples, responses

    
class LetterStatModel(object):
    class_n = 1
    train_ratio = 0.8

    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape
        new_samples = np.zeros((sample_n * self.class_n, var_n+1), np.float32)
        new_samples[:,:-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:,-1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples

    def unroll_responses(self, responses):
        if self.class_n  == 1:
            return responses
        sample_n = len(responses)
        new_responses = np.zeros(sample_n*self.class_n, np.int32)
        resp_idx = np.int32( responses + np.arange(sample_n)*self.class_n )
        new_responses[resp_idx] = 1
        return new_responses

        
class MLP(LetterStatModel):
    def __init__(self):
        self.model = cv2.ml.ANN_MLP_create()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, self.class_n)
        layer_sizes = np.int32([var_n, 150, 100, 50, self.class_n])

        self.model.setLayerSizes(layer_sizes)
        self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        self.model.setBackpropMomentumScale(0.0)
        self.model.setBackpropWeightScale(0.01)
        self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

        self.model.train(samples, cv2.ml.ROW_SAMPLE, np.float32(new_responses))

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)
        
        
class Boost(LetterStatModel):
    def __init__(self):
        self.model = cv2.ml.Boost_create()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_samples = self.unroll_samples(samples)
        new_responses = self.unroll_responses(responses)
        var_types = np.array([cv2.ml.VAR_NUMERICAL] * var_n + [cv2.ml.VAR_CATEGORICAL, cv2.ml.VAR_CATEGORICAL], np.uint8)

        self.model.setWeakCount(70)
        self.model.setMaxDepth(8)
        self.model.setMinSampleCount(10)
        self.model.setWeightTrimRate(0.00)
        self.model.setBoostType(cv2.ml.BOOST_GENTLE)
        self.model.train(cv2.ml.TrainData_create(new_samples, cv2.ml.ROW_SAMPLE, new_responses.astype(int), varType = var_types))

    def predict(self, samples):
        new_samples = self.unroll_samples(samples)
        ret, resp = self.model.predict(new_samples)
        return resp.ravel().astype(np.int)
        
        
def train_boost(csv_file, outfile):
    model = Boost()
    samples, responses = load_base(csv_file)
    
    char = np.where(responses == 0)[0]
    nochar = np.where(responses != 0)[0]

    test_n = int(min(len(char), len(nochar))*(1.0 - model.train_ratio))
    
    test_samples = np.random.choice(char, test_n)
    test_samples = np.append(test_samples, np.random.choice(nochar, test_n))
    train_samples = np.delete(np.arange(len(samples)), test_samples)
    
    model.train(samples[train_samples], responses[train_samples])
    
    print('testing...')
    
    train_rate = np.mean(model.predict(samples[train_samples]) == responses[train_samples].astype(int))
    test_rate  = np.mean(model.predict(samples[test_samples]) == responses[test_samples].astype(int))

    print('train rate: %f  test rate: %f' % (train_rate*100, test_rate*100))
    
    model.save(outfile)
    
    
if not path.exists(DATA_PATH):
    print("%s not found, start croping"%(DATA_PATH))
    system("python3 %s"%(path.join(main_dir,"marking_tools/crop_pricers.py")))
    print("croping done")
else:
    print("%s found, skip croping. To force restart delete this folder"
          %(DATA_PATH))
    
if not path.exists(DATASET_PATH):
    print("%s not found, start croping"%(DATASET_PATH))
    flush_db(DATA_PATH, DATASET_NAME, normalise=True)
    print("croping done")
else:
    print("%s found, skip croping. To force restart delete this folder"
          %(DATASET_PATH))

if not path.exists(DATASET_NEGATIVE_PATH):
    print("%s not found, start croping"%(DATASET_NEGATIVE_PATH))
    flush_db_dupl(DATA_PATH, DATASET_NEGATIVE_NAME, normalise=True)
    print("croping done")
else:
    print("%s found, skip croping. To force restart delete this folder"
          %(DATASET_NEGATIVE_PATH))
    
if not path.exists(CLASSIFIER_NM1_BIN):
    print("%s not found, start make"%(CLASSIFIER_NM1_BIN))
    system("python3 %s"%(path.join(main_dir,"symbol_split_utils/"
                                            "nm_dataset_builder/deploy.py")))
    print("make done")
else:
    print("%s found, skip make. To force make delete this folder"
          %(CLASSIFIER_NM1_BIN))
    
if not path.exists(NM1_CVS):
    print("%s not found, start converting"%(NM1_CVS))
    command = "python3 %s %s %s --out=%s"%(path.join(CLASSIFIER_DIR,
                                                     "bin/"
                                                     "build_csv_datasetNM1.py"),
                                           DATASET_PATH,
                                           DATASET_NEGATIVE_PATH,
                                           NM1_CVS
                                           )
    system(command)
else:
    print("%s found, skip converting. To force converting delete this folder"
          %(NM1_CVS))
    
if not path.exists(NM2_CVS):   
    command = "python3 %s %s %s --out=%s"%(path.join(CLASSIFIER_DIR,
                                                     "bin/",
                                                     "build_csv_datasetNM2.py"),
                                           DATASET_PATH,
                                           DATASET_NEGATIVE_PATH,
                                           NM2_CVS
                                           )
    system(command)
    print("converting done")
else:
    print("%s found, skip converting. To force converting delete this folder"
          %(NM2_CVS))

    

train_boost(NM1_CVS, CLASSIFIER_NM1_PATH)
train_boost(NM2_CVS, CLASSIFIER_NM2_PATH)
