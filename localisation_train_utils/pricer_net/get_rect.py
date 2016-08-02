#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 20:44:48 2016

@author: chernov
"""

from pprint import pprint
from collections import namedtuple
from argparse import ArgumentParser

import caffe
import cv2
import numpy as np

RESIZE = (250, 500)
Model = namedtuple('Model', ['struct', 'weights'])

def draw_point(img, point):
    point = (point*RESIZE).astype(np.int).ravel()
    cv2.circle(img, tuple(point), 3, (0,0,255), 2)

    return img

    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('image')
    return parser.parse_args()

    
if __name__ == '__main__': 
    args = parse_args()
        
    model = Model('pricer_net.prototxt', 'pricer_iter_100000.caffemodel')
    net = caffe.Net(model.struct, model.weights, caffe.TEST)
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    
    color_img = cv2.imread(args.image)
    raw_img = cv2.imread(args.image, 0)
    raw_img = cv2.normalize(raw_img.astype('float'), None, 0.0, 1.0,
                            cv2.NORM_MINMAX)
    image = np.empty(raw_img.shape + (1,), dtype=raw_img.dtype)

    for y in range(raw_img.shape[0]):
          for x in range(raw_img.shape[1]): 
                image[y][x] = [raw_img[y][x]]

    transformed_image = transformer.preprocess('data', image)
    
    net.blobs['data'].data[...] = transformed_image
    res = net.forward()["ip2"]
    
    image = cv2.imread(args.image)
    point = res.reshape((1,2))
    image = cv2.resize(image, RESIZE)
    image = draw_point(image, point)

    cv2.imshow("rect", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
