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
        if len(image.shape, ) == 2:
            img_buf = np.zeros(image.shape + (3,), image.dtype)
            for i in range(0,3):
                img_buf[:,:,i] = image
            image = img_buf
        
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
        