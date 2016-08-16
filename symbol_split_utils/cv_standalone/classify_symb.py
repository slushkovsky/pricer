#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 20:44:48 2016

@author: chernov
"""

import json

import numpy as np
import caffe
from scipy.signal import argrelextrema
    
class SymbolsClassifier():
    def __init__(self, struct_file, weights_file, dict_file):
        with open(dict_file) as dict_:
            self.symbol_dict = json.load(dict_)
            
        self.net = caffe.Net(struct_file, weights_file, caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': \
                                                 self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 1, 0))
        self.transformer.set_raw_scale('data', 255)
        
    def predict(self, image):
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image
        res = self.net.forward()["loss"]
        prob = self.symbol_dict[str(res.argmax())]
        return prob
        
        
class PriceClassifier(SymbolsClassifier):
    def covert_rects_to_price(self, img, rects):
        rubles = ""
        rects = sorted(rects, key=lambda x: x[0])
        for symbol in rects:
                sym = self.predict(img[symbol[1]: symbol[1] + symbol[3], 
                                       symbol[0]: symbol[0] + symbol[2]])
                rubles += sym[0]
        return int(rubles)
        

class NameClassifier(SymbolsClassifier):
    def covert_rects_to_text(self, img, rects,
                             conv_symb_h_ratio=2,
                             symb_space_w_ratio=2):
        if len(rects) == 0:
            return ""
        
        symbols = np.zeros(img.shape[0:2], np.uint8)
        for rect in rects:
            symbols[rect[1]: rect[1] + rect[3],
                    rect[0]: rect[0] + rect[2]] = 255
        hist = symbols.sum(axis=1)
        mean_symb_h = int(np.mean(rects, axis=0)[3])
        hist = np.convolve(hist, np.full((mean_symb_h//conv_symb_h_ratio),
                                         1, np.int64),
                           'same')
        
        lines = argrelextrema(hist, np.greater_equal,
                              order=mean_symb_h//conv_symb_h_ratio)[0]
        lines = lines[hist[lines]>0]
        lines = [line for i, line in enumerate(lines) \
                 if i == len(lines) - 1 or not\
                 (hist[line] == hist[lines[i + 1]] and
                  lines[i + 1] - line < mean_symb_h)]       
        name_string = ""
        for line in lines:
            rects_line = [rect for rect in rects \
                          if abs((rect[1] + rect[3]/2) - line) < mean_symb_h]
            mean_symb_width = int(np.mean(rects_line, axis=0)[2])                   
                                  
            rects_line = sorted(rects_line, key=lambda x: x[0])
            for i, symbol in enumerate(rects_line):
                sym = self.predict(img[symbol[1]: \
                                             symbol[1] + symbol[3],
                                             symbol[0]: \
                                             symbol[0] + symbol[2]])
                name_string += sym[0].lower()
                if i + 1 < len(rects_line):
                    dist_to_next = abs(symbol[0] + symbol[2] - 
                                       rects_line[i + 1][0])
                    if dist_to_next > mean_symb_width//symb_space_w_ratio:
                        name_string += " "                               
            name_string += "\n"
        return name_string
