#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 20:44:48 2016

@author: chernov
"""

import sys
import json
from os import path

main_dir = path.abspath(path.join(path.dirname(__file__), "../.."))
if not main_dir in sys.path:
    sys.path.append(main_dir)
del main_dir

import numpy as np
import cv2
import caffe
from scipy.signal import argrelextrema

from utils.os_utils import show_hist


    
class SymbolsClassifier():
    def __init__(self, struct_file, weights_file, dict_file,
                 out_layer_name="loss"):
        with open(dict_file) as dict_:
            self.symbol_dict = json.load(dict_)
            
        self.net = caffe.Net(struct_file, weights_file, caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': \
                                                 self.net.blobs['data'].data.shape})
        
        shape = self.net.blobs['data'].shape
        if shape[1] == 3:
            self.transformer.set_transpose('data', (2, 1, 0))
        elif shape[1] == 1:
            self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_raw_scale('data', 255)
        self.out_layer_name = out_layer_name
        
    def predict(self, image):
        transformed_image = None
        shape = self.net.blobs['data'].shape
        if shape[1] == 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = gray.swapaxes(1,0)
            image  = np.empty(gray.shape + (1,), dtype=gray.dtype)
            for y in range(gray.shape[0]):
                  for x in range(gray.shape[1]): 
                        image[y][x] = [gray[y][x]]
         
        transformed_image = self.transformer.preprocess('data', image)
            
        self.net.blobs['data'].data[...] = transformed_image
        res = self.net.forward()[self.out_layer_name]
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
                             symb_space_w_ratio=2,
                             w_perc=0.8, std_perc=0.6):
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
        #сливание близких строк
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
            
            distances = np.array(())
            for i in range(len(rects_line) - 1):
                x1 = max(rects_line[i][0], rects_line[i][0] + rects_line[i][2])
                x2 = min(rects_line[i + 1][0], rects_line[i + 1][0] + rects_line[i + 1][2])
                distances = np.append(distances, x2 - x1)
                
            hist, bins = np.histogram(distances, 8)
            #show_hist(hist, bins)
            hist = np.cumsum(hist[::-1])[::-1]
            #show_hist(hist, bins)
            
            words_approx = len(rects_line) / 7.2
            treshold = np.argmax(hist < words_approx)
            idx = np.argmax(hist == hist[treshold])
            if bins[idx - 1] - bins[idx - 1] < words_approx / 3.0:
                idx -= 1
            treshold = bins[idx]
            treshold = max(mean_symb_width//3, treshold)
            #print(treshold)
            #show_hist(distances, np.arange(len(distances) + 1))
            
            variances = np.zeros((len(rects_line)))
            
            for i, symbol in enumerate(rects_line):
                symb_img = img[symbol[1]: symbol[1] + symbol[3],
                               symbol[0]: symbol[0] + symbol[2]]
                variances[i] += symb_img.var()
           
            mean_var = np.mean(variances)
            symb_vars_disp = np.sort(np.abs(variances[:] - mean_var))
            var_thr = symb_vars_disp[int(len(symb_vars_disp)*w_perc)]  
            
            mean_var = variances[symb_vars_disp < var_thr].mean()
            std_var = variances[symb_vars_disp < var_thr].std()
            
            for i, symbol in enumerate(rects_line):
                symb_img = img[symbol[1]: symbol[1] + symbol[3],
                               symbol[0]: symbol[0] + symbol[2]]
                
                sym = self.predict(symb_img)
                name_string += sym[0].lower()
                if i + 1 < len(rects_line):
                    if distances[i] > treshold:
                        if symb_img.var() < mean_var + std_var*std_perc:
                            name_string += " "
                        else:
                            name_string += "*"                              
            name_string += "\n"
        return name_string
