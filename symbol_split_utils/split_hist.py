#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 22:18:37 2016

@author: chernov
"""

import os
from os import path
import sys
from argparse import ArgumentParser

main_dir = path.dirname(path.dirname(__file__))
if not main_dir in sys.path:
    sys.path.append(main_dir)

import numpy as np
from scipy.signal import argrelextrema
import cv2

from marking_tools.os_utils import show_hist

def extract_regions_from_hist(hist, treshold):
    
    if not hasattr(treshold, '__iter__'):
        th = np.full(len(hist), treshold)
    else:
        th = np.array(treshold)
    
    cur = hist[0] > th[0]
    parts = []

    y_up = -1
    if cur:
        y_up = 0
        
    for i in range(len(hist) - 1):
        if cur:
            if hist[i] > th[i] and hist[i+1] < th[i + 1]:
                parts.append([y_up, i])
                cur = False
                continue
        if not cur:
            if hist[i] < th[i] and hist[i+1] > th[i + 1]:
                y_up = i
                cur = True
                
    if cur:
        parts.append([y_up, i])
        
    return parts
    
    
def crop_regions(parts, merge_treshold = 10):
    parts_merged = []
    for part in parts:
        if abs(part[1] - part[0]) > merge_treshold:
            parts_merged.append(part)
    return parts_merged
    
    
def split_lines_hist(img, test = False):
    
    TRESHOLD_COEF = 1.001
    CONVOLUTION_KERNEL_SIZE = 0.05
    MERGE_TRESHOLD = 0.15
    EXTREMUM_ORDER = 0.125
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    binary = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    
    hist = binary.sum(axis=1)
    hist = hist/hist.max()
    if test:
        show_hist(hist, np.arange(len(hist) + 1))
    
    conv_kern_size = int(len(hist) * CONVOLUTION_KERNEL_SIZE)
    conv_kern = np.ones(conv_kern_size)
    hist_sm = np.convolve(hist, conv_kern, 'same')
    hist_sm = hist_sm/hist_sm.max()
    
    if test:
        show_hist(hist_sm, np.arange(len(hist_sm) + 1))
    
    order = int(len(hist_sm) * EXTREMUM_ORDER)
    
    minima = argrelextrema(hist_sm, np.less, order=order)[0]
    maxima = argrelextrema(hist_sm, np.greater, order=order)[0]
    
    maxima = np.insert(maxima, 0, 0)
    maxima = np.insert(maxima, len(maxima), len(hist))
    
    var_global = hist_sm.var()
    mean_global = hist_sm.mean()
    
    if test:
        print("maxima: ", maxima)
        print("minima: ", minima)
        print(var_global, mean_global)
            
    for i in range(len(maxima) - 1):
        var = hist_sm[maxima[i]:maxima[i + 1]].var()
        mean = hist_sm[maxima[i]:maxima[i + 1]].mean()
        coef = var/mean*mean_global/var_global
        if coef < MERGE_TRESHOLD:
            min_inside = np.where(np.logical_and(minima >= maxima[i],
                                                 minima <= maxima[i + 1]))[0]
            minima = np.delete(minima, min_inside)
    
    th = None
    if len(minima) != 0:
        th = np.zeros(len(hist))
        for i in range(len(maxima) - 1):
            range_l = i
            range_u = i+1
            tick = True
            while(len(np.where(np.logical_and(minima >= maxima[range_l],
                                              minima <= maxima[range_u]))[0]) == 0):
                if tick:
                    range_l = max(0, range_l - 1)
                else:
                    range_u = min(len(maxima) - 1, range_u + 1)
                tick = not tick
                
            th_cur = minima[np.where(np.logical_and(minima >= maxima[range_l],
                                                    minima <= maxima[range_u]))].min()
            th[maxima[i]: maxima[i + 1]] = hist_sm[th_cur] * TRESHOLD_COEF
    else:
        hist_min = hist_sm.min()
        if hist_min:
            th = hist_sm.min() * TRESHOLD_COEF
        else:
            th = hist_sm.max() / 10
    
    parts = extract_regions_from_hist(hist_sm, th)
    return parts
    

def split_symbols_list(gray, treshold, treshold_dark):
    if not (gray.shape[0] and gray.shape[1]):
        return []
    
    binary = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    binary = ((binary/binary.max())*255).astype(np.uint8)
    ret, mask_light = cv2.threshold(gray, treshold_dark, 255,
                                        cv2.THRESH_BINARY_INV)
    binary = cv2.bitwise_and(binary, binary, mask=mask_light)
    
    hist_x = binary.sum(axis=0)
    hist_x = hist_x/hist_x.max()
    parts_x = extract_regions_from_hist(hist_x, treshold)
    return parts_x


def mask_hist(img, test=True):
    TRESHOLD = 0.23
    TRESHOLD_X = 0.12
    DARK_TRESHOLD = 180
    
    cv2.namedWindow("masked_hor")
    cv2.createTrackbar("th_x", "masked_hor", 0, 100, lambda x: None)
    cv2.setTrackbarPos("th_x", "masked_hor", int(TRESHOLD_X * 100))
    cv2.createTrackbar("th_dark", "masked_hor", 0, 255, lambda x: None)
    cv2.setTrackbarPos("th_dark", "masked_hor", DARK_TRESHOLD)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    while True:
        treshold_x = cv2.getTrackbarPos("th_x", "masked_hor") / 100
        treshold_dark = cv2.getTrackbarPos("th_dark", "masked_hor")
        
        parts = split_lines_hist(img)
        mask = np.zeros(gray.shape, np.uint8)
        for part in parts:
            mask[part[0]:part[1], :] = 255
    
        
        for part in parts:
            part_img = gray[part[0]:part[1],:]
            parts_x = split_symbols_list(part_img, treshold_x, treshold_dark)
            mask_line = np.zeros((part[1] - part[0], gray.shape[1]), np.uint8)
            for part_x in parts_x:
                mask_line[:,part_x[0]:part_x[1]] = 255
            mask[part[0]:part[1],:] = mask_line 
                
        masked_hor = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("masked_hor", masked_hor)
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        
        
def parse_args():
    def file_arg(value): 
        if not os.path.exists(value):
            if not os.path.exists(value):
                raise Exception() 
        return value
        
    parser = ArgumentParser()
    parser.add_argument('path', type=file_arg,
                        help="Image filepath or path to folder with images")
    return parser.parse_args()
    
    
if __name__ == '__main__':
    IMG_H = 400
    
    args = parse_args()
    
    path = args.path
    imgs = None
    folder = None
    if os.path.isdir(path):
        imgs = os.listdir(path)
        folder = path
    else:
        imgs = [os.path.basename(path)]
        folder = os.path.dirname(os.path.realpath(path))
    
    for img_name in imgs:
        img_path = os.path.join(folder,img_name)
        print(img_path)
        img = cv2.imread(img_path)
        if img.shape[0] > IMG_H:
            w = int(IMG_H*img.shape[1]/img.shape[0])
            img = cv2.resize(img, (w, IMG_H), cv2.INTER_CUBIC)
        
        mask_hist(img)
        k = cv2.waitKey() & 0xFF
        if k == 27:
            break
        
    cv2.destroyAllWindows()
    