#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 20:44:48 2016

@author: chernov
"""

import copy
from pprint import pprint
from collections import namedtuple
from argparse import ArgumentParser

import caffe
import cv2
import numpy as np

RESIZE = (500, 250)
Model = namedtuple('Model', ['struct', 'weights'])

def draw_contour(img, contour):
    contour = copy.copy(contour)
    shape = img.shape
    
    for i in range(0, len(contour), 2):
        contour[i] *= shape[1]
        contour[i + 1] *= shape[0] 

    contour = contour.reshape((4,2)).astype(np.int)
    cv2.drawContours(img, [contour], 0, (0, 0, 255),
                     shape[0] // 100)
    return img

    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('image')
    return parser.parse_args()

    
if __name__ == '__main__': 
    args = parse_args()
        
    model = Model('rubli_net.prototxt', 'rubli_iter_30000.caffemodel')
    net = caffe.Net(model.struct, model.weights, caffe.TEST)
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 1)
    
    image = caffe.io.load_image(args.image)
    transformed_image = transformer.preprocess('data', image)
    
    net.blobs['data'].data[...] = transformed_image
    res = net.forward()["ip2"]
    
    image = cv2.imread(args.image)
    image = draw_contour(image, res[0])
    image = cv2.resize(image, RESIZE)
    pprint(((res.reshape((4,2)))*RESIZE).astype(np.int))
    cv2.imshow("rect", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
