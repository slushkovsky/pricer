#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 23:51:24 2016

@author: chernov
"""

from argparse import ArgumentParser

import cv2
import numpy as np

def process_image(img):
    MIN_H_MASK = 0.15
    MIN_SYMB_W = 0.01
    MIN_SYMB_H = 0.4
    TRESH_STEPS = 6
    MIN_VARIANCE = 1000
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    step = img.shape[1]//TRESH_STEPS
    otsu = np.zeros(gray.shape, np.uint8)
    for i in range(0, img.shape[1], step):
        ret2, otsu[:, i:i+step] = cv2.threshold(gray[:, i:i+step], 
                                                0, 255, cv2.THRESH_OTSU +
                                                cv2.THRESH_BINARY_INV)
    
    otsu_hist = otsu.sum(axis=0)
    otsu_hist = otsu_hist/otsu_hist.max()
    otsu[:, otsu_hist < MIN_H_MASK ] = 0 

    im2, contours, hierarchy = cv2.findContours(otsu, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    
    contours_out = []
    for i in range(0, len(contours)):
        if hierarchy[0][i][3] == -1:
            contours_out.append(contours[i])
    
    rects = []
    rects_bad = []        
    for contour in contours_out:
        x, y, w, h = cv2.boundingRect(contour)
        
        if gray[y: y+h, max(0, x - w//5):
                min(gray.shape[1], x + int(w*6/5))].var() < MIN_VARIANCE:
            rects_bad.append([x,y,w,h])
            continue
        
        if (w > img.shape[1]*MIN_SYMB_W and
            h > img.shape[0]*MIN_SYMB_H):
            rects.append([x, y, w, h])
        else:
            rects_bad.append([x,y,w,h])
            
    rects = np.array(rects)
    if len(rects):
        rects[:, 1] = rects[:, 1].min()
        rects[:, 3] = rects[:, 3].max()
        rects = tuple(rects)  
    return rects


def parse_args():   
    parser = ArgumentParser()
    parser.add_argument('image', type=str,
                        help="Image filepath")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
        
    img = cv2.imread(args.image)
    rects =  process_image(img)
    
    vis = img.copy()
    for rect in rects:
        cv2.rectangle(vis, (rect[0], rect[1]),
                      (rect[0] + rect[2], rect[1] + rect[3]),
                      (0, 0, 255), 2)

    cv2.imshow("vis", vis)
    cv2.waitKey()
    cv2.destroyAllWindows()