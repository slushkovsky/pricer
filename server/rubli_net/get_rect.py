#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 20:44:48 2016

@author: chernov
"""

import os
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
        
    model = Model('rubli_net.prototxt', 'pricer_rubli2_iter_100000.caffemodel')
    net = caffe.Net(model.struct, model.weights, caffe.TEST)
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 1)
    
    images = None
    if os.path.isdir(args.image):
        images = [f for f in os.listdir(args.image) if f.endswith('.jpg')]
    else:
        images = [args.image]

    outfile = None
    if len(images) != 1:
        outfile = open(os.path.join(args.image, "points.txt"), "w")

    for image_name in images:   
        image_path = os.path.join(args.image, image_name)
        color_img = cv2.imread(image_path)
        raw_img = cv2.imread(image_path, 0)
        raw_img = cv2.normalize(raw_img.astype('float'), None, 0.0, 1.0,
                                cv2.NORM_MINMAX)
        image = np.empty(raw_img.shape + (1,), dtype=raw_img.dtype)
    
        for y in range(raw_img.shape[0]):
              for x in range(raw_img.shape[1]): 
                    image[y][x] = [raw_img[y][x]]
    
        transformed_image = transformer.preprocess('data', image)
        
        net.blobs['data'].data[...] = transformed_image
        res = net.forward()["ip2"]
        
        if len(images) == 1:
            image = cv2.imread(image_path)
            image = draw_contour(image, res[0])
            image = cv2.resize(image, RESIZE)
            pprint(((res.reshape((4,2)))*RESIZE).astype(np.int))
            cv2.imshow("rect", image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            contour = res.reshape((4,2))
            for i in range(0, len(contour), 2):
                contour[i] *= image.shape[1]
                contour[i + 1] *= image.shape[0] 
            
            outfile.write(image_name)
            for point in contour.astype(np.int).ravel():
                outfile.write(" %s"%(point))
            outfile.write("\n")
