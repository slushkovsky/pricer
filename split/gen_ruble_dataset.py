#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 00:04:29 2016

@author: chernov
"""

from argparse import ArgumentParser
from os import path, listdir, makedirs
import shutil

import cv2

from split_ruble_symbols import process_image

def parse_args():   
    parser = ArgumentParser()
    parser.add_argument('data', type=str,
                        help="Dataset path (folder with images)")
    parser.add_argument('outpath', type=str,
                        help="Output folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if(path.exists(args.outpath)):
        shutil.rmtree(args.outpath)
    makedirs(args.outpath)
    
    images = [f for f in listdir(args.data) if f.endswith('.jpg')]
    idx = 0
    for image in images:
        img = cv2.imread(path.join(args.data, image))
        rects =  process_image(img)
        
        for rect in rects:
            symb = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :]
            cv2.imwrite(path.join(args.outpath, "%s.png"%(idx)), symb)
            idx += 1