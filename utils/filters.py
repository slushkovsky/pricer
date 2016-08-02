#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 15:18:34 2016

@author: chernov
"""

import copy

import sys
from os import path

pricer_dir = path.dirname(path.dirname(__file__))
if not pricer_dir in sys.path:
    sys.path.append(pricer_dir)
del pricer_dir

import cv2
import numpy as np

from utils.os_utils import show_hist
import symbol_split_utils.split_hist as split_utils

IMAGE_SIZE = (576, 1024)

def filter_dark_gray(color, test=False):
    
    hsv = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    
    s_norm = cv2.normalize(hsv[:,:,1], None, alpha=0.0, beta=1.0,
                           norm_type=cv2.NORM_MINMAX,
                           dtype = cv2.CV_64F)
    gray_norm = cv2.normalize(hsv[:,:,2], None, alpha=0.0, beta=1.0,
                              norm_type=cv2.NORM_MINMAX,
                              dtype = cv2.CV_64F)
    
    if test:
        cv2.imshow("1.0 - gray_norm", 1.0 - gray_norm)
        cv2.imshow("1.0 - s_norm", 1.0 - s_norm)
    
    mask_dark_gray = ((1.0 - gray_norm)**2) * (1.0 - s_norm)
    mask_dark_gray = cv2.normalize(mask_dark_gray, None, alpha=0, beta=255,
                                   norm_type=cv2.NORM_MINMAX,
                                   dtype = cv2.CV_8U)
    
    if test:
        cv2.imshow("mask_dark_gray", mask_dark_gray)
    return mask_dark_gray

    
def mask_full_hor_lines(img, test=False):
    LINE_THICKNESS = 40
    Y_DISP_PIXELS = 50
    DISP_STEPS = 20
    SOBEL_KERNEL_SIZE = 3
    GRADIENT_MIN = 0.5
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
    sobelx = np.abs(sobelx)
    sobelx = cv2.normalize(sobelx, None, alpha=0.0, beta=1.0, 
                           norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    if test:
        cv2.imshow('mask_full_hor_lines sobelx', sobelx)
        
    #dark_gray_mask = filter_dark_gray(img, test)
    #kernel = np.ones((5,5),np.uint8)
    #dil = cv2.dilate(dark_gray_mask, kernel, iterations=1)
    #sobel_gray = cv2.normalize( sobelx * dil, None, alpha=0.0, beta=1.0, 
    #                           norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    #cv2.imshow("sobel_gray", sobel_gray)
    
    #hist = sobelx.sum(axis=1)
    #hist = hist/hist.max()
    #if test:
    #    hist_3 = np.convolve(hist, np.ones(100), 'same')
    #    show_hist(hist_3, np.arange(len(hist_3) + 1))
        
    full_mask = np.zeros(gray.shape, np.uint8)
    for y in range(0, sobelx.shape[0], LINE_THICKNESS):
       for y_left in range(y - Y_DISP_PIXELS//2,
                           y + Y_DISP_PIXELS//2,
                           Y_DISP_PIXELS//(DISP_STEPS//2)):
           mask = np.zeros(gray.shape, np.uint8)
           cv2.line(mask, (0, y), (sobelx.shape[1], y_left), 255, LINE_THICKNESS)
           masked = cv2.bitwise_and(sobelx, sobelx, mask=mask)
           if(not (masked > GRADIENT_MIN).sum() > img.shape[1]*0.05):
               full_mask = cv2.bitwise_or(full_mask, mask)        
    full_mask = 255 - full_mask
    if test:
        cv2.imshow("mask_full_hor_lines mask", full_mask)  
    return full_mask

    
def monochrome_mask(color, max_dispersion=10, min_intensity=0,
                    max_intensity=255):
    mask = np.zeros((color.shape[0], color.shape[1]), np.uint8)
    for intensity in range(min_intensity, max_intensity, max_dispersion):
        gray_cur = np.full(3, intensity, np.uint8)
        mask_part = cv2.inRange(color, gray_cur - max_dispersion,
                                gray_cur + max_dispersion)
        mask = cv2.bitwise_or(mask, mask_part)
    return mask


def monochrome_filter(color, max_dispersion=10, min_intensity=0,
                      max_intensity=255):
    filtered = copy.copy(color)
    mask = monochrome_mask(color, max_dispersion, min_intensity, max_intensity)
    masked = cv2.bitwise_and(filtered,filtered, mask=mask)
    return masked


def pricer_background_mask(color, central_crop_r=40, black_max_intensity=150,
                           backgr_bandwidth=80, dispersion_rel = 0.3):    
    shape = color.shape
    center = color[shape[0]//2 - central_crop_r: shape[0]//2 + central_crop_r, 
                   shape[1]//2 - central_crop_r: shape[1]//2 + central_crop_r]   
    #вычисление среднего цвета фона
    pixels = 0
    mean = np.array((0,0,0))
    for row in range(0, center.shape[0]):
        for col in range(0, center.shape[1]):
            if (center[row, col] > black_max_intensity).any():
                pixels += 1
                mean += center[row, col]
    mean = mean/pixels
    # верхняя граница увеличивается, чтобы захватить блики
    mask = np.zeros(color.shape[0:2], np.uint8)
    for intensity in range(-backgr_bandwidth//2,
                           backgr_bandwidth//2, 1):
        scale = 1 + dispersion_rel
        mask_part = cv2.inRange(color, (mean - intensity)/scale, 
                                (mean - intensity)*scale)
        mask = cv2.bitwise_or(mask, mask_part)
    return mask


def pricer_background_filter(color, central_crop_r=30, black_max_intensity=150,
                           backgr_bandwidth=80, dispersion_rel = 0.3):
    mask = pricer_background_mask(color, central_crop_r, black_max_intensity,
                                  backgr_bandwidth, dispersion_rel)
    filtered = cv2.bitwise_and(color,color, mask=mask)
    return filtered

    
if __name__ == '__main__':
    from os_utils import ask_image_path
    img = cv2.imread(ask_image_path())
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("original", img)
    filter_color = pricer_background_filter(img)
    cv2.imshow("pricer_background_filter", filter_color)
    filtered = monochrome_filter(img)
    cv2.imshow("monochrome_mask", filtered)
    cv2.waitKey()
    cv2.destroyAllWindows()
            