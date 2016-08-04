#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 20:42:17 2016

@author: chernov
"""

import copy
from os import path
from argparse import ArgumentParser

import cv2

from split_hist_cv import split_lines_hist, crop_regions
from split_symbols_cv import detect_text_cv

def parse_args():
    def file_arg(value): 
        if not path.exists(value):
            if not path.exists(value):
                raise Exception() 
        return value
        
    parser = ArgumentParser()
    parser.add_argument('image', type=file_arg,
                        help="Path to image with name")
    parser.add_argument('--nm1', type=file_arg,
                        help="Path to pretrained NM1 dtree classifier")
    parser.add_argument('--nm2', type=file_arg,
                        help="Path to pretrained NM2 dtree classifier")
    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_args()
    
    img = cv2.imread(args.image)
    regions = crop_regions(split_lines_hist(img),
                           img.shape[0]*0.15)
    
    nm1 = path.abspath(path.join(path.dirname(__file__), 
                                 "pretrained_classifiers/"
                                 "trained_classifierNM1.xml"))
    nm2 = path.abspath(path.join(path.dirname(__file__), 
                                 "pretrained_classifiers/"
                                 "trained_classifierNM2.xml"))
    
    if not args.nm1 is None:
        nm1 = args.nm1
    if not args.nm2 is None:
        nm2 = args.nm2

    rects, rects_bad = [], []
    
    for i in range(len(regions)):
        cur_img_part = img[regions[i][0]:regions[i][1],:]
        cur_rects, cur_rects_bad = detect_text_cv(cur_img_part,
                                                  nm1, 
                                                  nm2)
        
        for rect in cur_rects:
            rect[1] += regions[i][0]
            rects.append(rect)
            
        for rect in cur_rects_bad:
            rect[1] += regions[i][0]
            rects_bad.append(rect)
    
    vis = copy.copy(img)   
    for rect in rects:
        cv2.rectangle(vis, (rect[0], rect[1]),
                      (rect[0] + rect[2], rect[1] + rect[3]),
                      (0, 0, 255), 2)
    for rect in rects_bad:
        cv2.rectangle(vis, (rect[0], rect[1]),
                      (rect[0] + rect[2], rect[1] + rect[3]),
                      (0, 255, 255), 1)
    cv2.imshow("vis", vis)
    cv2.waitKey()
    cv2.destroyAllWindows()
