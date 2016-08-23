#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 13:56:42 2016

@author: chernov
"""

import json
from urllib.request import urlopen, Request
import base64
from argparse import ArgumentParser
from os import path, remove

import cv2
import numpy as np


def parse_args():
    def file_arg(value): 
        if not path.exists(value):
            if not path.exists(value):
                raise Exception() 
        return value
        
    parser = ArgumentParser()
    
    parser.add_argument('image', type=file_arg,
                        help="image path")
    parser.add_argument('--uri', default="http://127.0.0.1:5000/price", 
                        help="command uri")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    req = Request(args.uri)
    req.add_header('Content-Type', 'application/json')

    temp_img = "temp.png"
    cv2.imwrite(temp_img, cv2.imread(args.image, 0))
    
    with open(temp_img, "rb") as f:
        base64String = base64.b64encode(f.read())
    remove(temp_img)
        
    message = {"id": 1, "base64String": base64String.decode()}
    response = urlopen(req, json.dumps(message).encode("utf-8"))   
    
    print(json.loads(response.read().decode("utf-8")))
