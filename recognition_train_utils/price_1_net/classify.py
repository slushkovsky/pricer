#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 20:44:48 2016

@author: chernov
"""

import json
from pprint import pprint
from collections import namedtuple
from argparse import ArgumentParser

import caffe

Model = namedtuple('Model', ['struct', 'weights'])

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('image')

    return parser.parse_args()

if __name__ == '__main__': 
    args = parse_args()

    symbol_dict = None
    with open("price_1_lmdb_dict.json") as dict_file:
        symbol_dict = json.load(dict_file)
        
    model = Model('price_1_net.prototxt', 'price_1_iter_1000.caffemodel')
    net = caffe.Net(model.struct, model.weights, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255)

    image = caffe.io.load_image(args.image)
    print(image.shape)
    transformed_image = transformer.preprocess('data', image)
    
    net.blobs['data'].data[...] = transformed_image
    res = net.forward()["loss"]
    prob = symbol_dict[str(res.argmax())]
    pprint(prob)
