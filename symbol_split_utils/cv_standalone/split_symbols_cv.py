#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 23:54:39 2016

@author: chernov
"""

import cv2
import numpy as np

def merge_close_rects(rects, merge_step_pixels=20):
    if not len(rects):
        return []

    rects_merged = [rects[0]]
    for i in range(len(rects)):
        rect = rects[i]
        any_ = False
        for j in range(len(rects_merged)):
            if abs(rect[0] - rects_merged[j][0]) < merge_step_pixels and \
               abs(rect[1] - rects_merged[j][1]) < merge_step_pixels and \
               abs(rect[2] - rects_merged[j][2]) < merge_step_pixels and \
               abs(rect[3] - rects_merged[j][3]) < merge_step_pixels:
                   any_=True
                   rects_merged[j][0] = min(rect[0], rects_merged[j][0])
                   rects_merged[j][1] = min(rect[1], rects_merged[j][1])
                   rects_merged[j][2] = max(rect[2], rects_merged[j][2])
                   rects_merged[j][3] = max(rect[3], rects_merged[j][3])
                   break
        if not any_:
            rects_merged.append(rect)
    return rects_merged

    
def rescale_rects(rects, scale):
    for i in range(len(rects)):
        rects[i][0] = int(rects[i][0] * scale[1])
        rects[i][2] = int(rects[i][2] * scale[1])
        rects[i][1] = int(rects[i][1] * scale[0])
        rects[i][3] = int(rects[i][3] * scale[0])
        
    return rects


def filter_rects_wh(rects, max_w_h_ratio=1.5):
    rects_buf = []
    for i in range(len(rects)):
        rect = rects[i]
        if not rect[2]/rect[3] > max_w_h_ratio:
            rects_buf.append(rect)
    return rects_buf


def find_rects_inside(rects):
    rects_inside = []
    rects_outside = []
    for i in range(len(rects)):
        contains = False
        for j in range(len(rects)):
            rect_i = rects[i]
            rect_j = rects[j]
            if(rect_i != rect_j and
               rect_i[0] >= rect_j[0] and
               rect_i[1] >= rect_j[1] and
               rect_i[0] - rect_j[0] + rect_i[2] <= rect_j[2] and
               rect_i[1] - rect_j[1] + rect_i[3] <= rect_j[3]):
                contains = True
                break
        if not contains:
            rects_outside.append(rect_i)
        else:
            rects_inside.append(rect_i)
    
    return rects_outside, rects_inside
    
    
def detect_text_cv(img, nm1_path, nm2_path):
    MAX_W_H_RATIO = 1.5
    IMG_H = 100
    MERGE_STEP_PIXELS = 20
    w = int(IMG_H*img.shape[1]/img.shape[0])
    scale = np.array(img.shape)
    img = cv2.resize(img, (w, IMG_H))
    scale = scale/np.array(img.shape)
    
    channels = cv2.text.computeNMChannels(img)

    rects = []
    for channel in channels:
        erc1 = cv2.text.loadClassifierNM1(nm1_path)
        er1 = cv2.text.createERFilterNM1(erc1,
                                         20,
                                         0.001,
                                         0.05,
                                         0.8,
                                         True,
                                         0.4)
        erc2 = cv2.text.loadClassifierNM2(nm2_path)
        er2 = cv2.text.createERFilterNM2(erc2,0.5)
        regions = cv2.text.detectRegions(channel,er1, er2)
        for i in range(len(regions)):
            x,y,w,h = cv2.boundingRect(regions[i])
            rects.append([x,y,w,h])
    
    rects_merged = []
    rects_outside = []
    rects_inside = []
    if len(rects) > 0:
        rects_merged = merge_close_rects(rects, MERGE_STEP_PIXELS)
        rects_merged = filter_rects_wh(rects_merged, MAX_W_H_RATIO)        
        rects_outside, rects_inside = find_rects_inside(rects_merged)
      
    rects_outside = rescale_rects(rects_outside, scale)
    rects_inside = rescale_rects(rects_inside, scale)    
    return rects_outside, rects_inside
