#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 14:24:22 2016

@author: chernov
"""

from os import environ, path, system
import sys

import cv2
import numpy as np

main_dir = path.dirname(path.dirname(__file__))
if not main_dir in sys.path:
    sys.path.append(main_dir)

from marking_tools.markers_db_tools import flush_db

DATASET_NAME = "symbols"
DATA_PATH = environ["BEORGDATAGEN"] + "/CData_full"
DATASET_PATH = DATA_PATH + "/" + DATASET_NAME
CLASSIFIER_NM1_PATH = environ["BEORGDATA"] + "/cv_trained_classifiers/trained_classifierNM1.xml"
CLASSIFIER_NM2_PATH = environ["BEORGDATA"] + "/cv_trained_classifiers/trained_classifierNM2.xml"

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
    train_ratio = 0.7

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
   
        
class Boost(LetterStatModel):
    def __init__(self):
        self.model = cv2.ml.Boost_create()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_samples = self.unroll_samples(samples)
        new_responses = self.unroll_responses(responses)
        var_types = np.array([cv2.ml.VAR_NUMERICAL] * var_n + [cv2.ml.VAR_CATEGORICAL, cv2.ml.VAR_CATEGORICAL], np.uint8)

        self.model.setWeakCount(300)
        self.model.setMaxDepth(3)
        self.model.setMinSampleCount(10)
        self.model.setWeightTrimRate(0)
        self.model.train(cv2.ml.TrainData_create(new_samples, cv2.ml.ROW_SAMPLE, new_responses.astype(int), varType = var_types))

    def predict(self, samples):
        new_samples = self.unroll_samples(samples)
        ret, resp = self.model.predict(new_samples)

        return resp.ravel().reshape(-1, self.class_n).argmax(1)
        
        
def train_boost(csv_file, outfile):
    boost = Boost()
    samples, responses = load_base(csv_file)
    boost.train(samples, responses)
    boost.save(outfile)

    
if not path.exists(DATA_PATH):
    print("%s not found, start croping"%(DATA_PATH))
    system("python3 ../marking_tools/crop_pricers.py")
    print("croping done")
else:
    print("%s found, skip croping. To force restart delete this folder"
          %(DATA_PATH))
    
if not path.exists(DATASET_PATH):
    print("%s not found, start croping"%(DATASET_PATH))
    flush_db(DATA_PATH, DATASET_NAME)
    print("croping done")
else:
    print("%s found, skip croping. To force restart delete this folder"
          %(DATASET_PATH))
    
csv_file_nm1 = "/home/chernov/projects/erfilter_train/char_datasetNM1.csv"
train_boost(csv_file_nm1, CLASSIFIER_NM1_PATH)
csv_file_nm2 = "/home/chernov/projects/erfilter_train/char_datasetNM2.csv"
train_boost(csv_file_nm2, CLASSIFIER_NM2_PATH)

