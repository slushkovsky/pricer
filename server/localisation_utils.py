#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 13:02:33 2016

@author: chernov
"""

import caffe
import cv2
import numpy as np

class Localisator():
    def __init__(self, model_file, weights_file, raw_scale=1.0,
                 in_layer="data", out_layer="ip2"):
        
        self.in_layer = in_layer
        self.out_layer = out_layer
        
        self.net = caffe.Net(model_file, weights_file, caffe.TEST)
        self.transformer = caffe.io.Transformer({in_layer: self.net.blobs[in_layer].data.shape})
        self.transformer.set_transpose(in_layer, (2, 0, 1))
        self.transformer.set_raw_scale(in_layer, raw_scale)
        
        
    def predict(self, img):
        raw_img = cv2.normalize(img.astype('float'), None, 0.0, 1.0,
                            cv2.NORM_MINMAX)
        image = np.empty(raw_img.shape + (1,), dtype=raw_img.dtype)
        for y in range(raw_img.shape[0]):
              for x in range(raw_img.shape[1]): 
                    image[y][x] = [raw_img[y][x]]
    
        transformed_image = self.transformer.preprocess(self.in_layer, image)
            
        self.net.blobs[self.in_layer].data[...] = transformed_image
        cnt = self.net.forward()[self.out_layer][0]
        for i in range(0, len(cnt), 2):
            cnt[i] *= img.shape[1]
            cnt[i + 1] *= img.shape[0]
        return cnt.reshape((4,2))
        
def draw_contour(img, contour):
    contour = contour.copy()
    shape = img.shape

    contour = contour.reshape((4,2)).astype(np.int)
    cv2.drawContours(img, [contour], 0, (0, 0, 255),
                     shape[0] // 100)
    return img
