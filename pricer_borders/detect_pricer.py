#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 20:03:39 2016

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

from utils.geom_functions import find_min_rect
from utils.filters import IMAGE_SIZE, monochrome_filter, mask_full_hor_lines
from utils.filters import monochrome_mask, pricer_background_mask
from utils.os_utils import show_hist
from barcode.detect_barcode import detect_barcode

def prepare_image(img, new_size=IMAGE_SIZE, h_crop_perc=0.25):
    scale = np.array(img.shape)
    img = cv2.resize(img, new_size, interpolation = cv2.INTER_CUBIC)
    scale = scale/np.array(img.shape)
    offset = img.shape[0]*h_crop_perc
    top_crop = int(img.shape[0] * h_crop_perc)
    bottom_crop = int(img.shape[0]* (1 - h_crop_perc))
    img = img[top_crop: bottom_crop, 0: img.shape[1]] 
    return img, scale, offset
  
    
def hist_filter(img, min_bin_tresh, num_bins=30, range_=(1, 255), test=False):
    PEAKS_INTEGRAL_RATIO = 1.4
    PEAKS_INTEGRAL_RATIO_REVERSED = 0.05
    PEAK_AREA = 0.333
    PEAK_WIDTH = 0.166
    
    peak_area_inv = int(PEAK_AREA**(-1.0))
    peak_width_inv = int(PEAK_WIDTH**(-1.0))
    hist, bins = np.histogram(img.ravel(), num_bins, range_)
    if test: 
        cv2.imshow("masked", img)
        show_hist(hist, bins)
    left = np.argmax(hist > min_bin_tresh)
    right = len(hist) - np.argmin(hist > min_bin_tresh)
    width = right - left
    l_peak = hist[left : left + width // peak_area_inv]
    r_peak = hist[(right - width // peak_area_inv) - width // peak_width_inv:
                  right  - width // peak_width_inv]
    max1, integral1 = (l_peak.max(), l_peak.sum())
    max2, integral2 = (r_peak.max(), r_peak.sum())
    if not (integral1 and (max2 > max1) and 
            (integral2 / integral1 > PEAKS_INTEGRAL_RATIO) and
            (integral1 / integral2 > PEAKS_INTEGRAL_RATIO_REVERSED)):
        if test: print("histogramm check failed")
        return False
    return True
    
    
def find_first_n_contours(gray, num=5, tresh_block_size=131,
                          tresh_c=2): 
    thresh = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,
                                   tresh_block_size,
                                   tresh_c)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    contours = contours[0:num]
    return contours
    

def apply_rect_mask(gray, x1, y1, w, h):
    rect_mask = np.zeros(gray.shape,np.uint8)
    cv2.rectangle(rect_mask,
                  (x1,y1),
                  (x1 + w, y1 + h),
                  color=(255,255,255), thickness=-1)
    masked = cv2.bitwise_and(gray, gray, mask=rect_mask)
    return masked
    
    
def detect_pricer(img): 
    MASK_DILATE_KERNEL_SIZE = 15
    PRICER_W_H_RATIO_MIN = 1.5
    PRICER_W_H_RATIO_MAX = 3.0
    MAX_DISPERSION = 10
    HIST_BIN_MIN_RELATIVE = 0.005
    
    img, scale, offset = prepare_image(img)
    monochrome_filtered = monochrome_filter(img, max_dispersion=MAX_DISPERSION)
    gray = cv2.cvtColor(monochrome_filtered,cv2.COLOR_BGR2GRAY)
    
    contours = find_first_n_contours(gray)
    passed = np.empty((0,4))
    if len(contours) == 0:
        return passed
    for i in range(0, len(contours)):
        mask = np.zeros(gray.shape,np.uint8)
        cv2.drawContours(mask,[contours[i]],0,255,-1)
        kernel = np.ones((MASK_DILATE_KERNEL_SIZE,MASK_DILATE_KERNEL_SIZE),
                         np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        (x1, y1, w, h) = find_min_rect(mask_dilated)
        
        if not (PRICER_W_H_RATIO_MIN <= w/h <= PRICER_W_H_RATIO_MAX):
            continue
        rect_masked = apply_rect_mask(gray, x1, y1, w, h)
        if(hist_filter(rect_masked, (w*h)*HIST_BIN_MIN_RELATIVE)):
            continue
        passed = np.vstack((passed, (x1 * scale[1],
                                    (y1 + offset) * scale[0], 
                                    w * scale[1],
                                    h * scale[0])))
    return passed
    
    
def detect_pricer_test(img, test=False):
    PRICER_W_H_RATIO_MIN = 1.5
    PRICER_W_H_RATIO_MAX = 3.0
    
    PRICER_H_W_RATIO_MIN = 0.5
    PRICER_H_W_RATIO_MAX = 3.0
    
    PRICER_W_MIN_PERCENT = 0.5
    PRICER_H_MIN_PERCENT = 0.2
    
    BARCODE_W_H_MIN = 1.5
    BARCODE_W_H_MAX = 4.0
    
    HIST_BIN_MIN_RELATIVE = 0.005
    
    img, scale, offset = prepare_image(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bgr_filter_mask = pricer_background_mask(img)
    
    lines_mask = mask_full_hor_lines(img, test)
    
    kernel = np.ones((5,5),np.uint8)
    bgr_filter_mask = cv2.dilate(bgr_filter_mask, kernel, iterations=1)   
    if test:
        masked = cv2.bitwise_and(gray,gray, mask=bgr_filter_mask)
        cv2.imshow("bgr_filter_mask", masked)
    
    mono_mask = monochrome_mask(img, max_dispersion=20, max_intensity=120)
    kernel = np.ones((5,5),np.uint8)
    mono_mask = cv2.dilate(mono_mask, kernel, iterations=1)
    if test:
        masked = cv2.bitwise_and(gray,gray, mask=mono_mask)
        cv2.imshow("mono_mask", masked)
    
    full_mask = cv2.bitwise_or(bgr_filter_mask, mono_mask)
    full_mask = cv2.bitwise_and(full_mask, lines_mask)
    
    if test:
        #masked = cv2.bitwise_and(gray,gray, mask=full_mask)
        cv2.imshow("full_mask", full_mask)
    
    contours = find_first_n_contours(full_mask)
    
    if test:
        draw_frame = copy.copy(img)
        cv2.drawContours(draw_frame, contours, -1,
                         (0,255,0), cv2.CHAIN_APPROX_TC89_KCOS)
        cv2.imshow("contours", draw_frame)
    
        
    passed = np.empty((0,4))
    for i in range(0, len(contours)):
        mask = np.zeros(gray.shape,np.uint8)
        cv2.drawContours(mask,[contours[i]],0,255,-1)
        #удаление мелких деталей внутри контура
        kernel = np.ones((15, 30), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        
        (x1, y1, w, h) = find_min_rect(mask_dilated)
        
        if not((x1 <= img.shape[1]//2 <= x1 + w) and
               (y1 <= img.shape[0]//2 <= y1 + h)):
            if test: print("rect not contains center")
            continue
        
        if not((PRICER_W_H_RATIO_MIN <= w/h <= PRICER_W_H_RATIO_MAX) or
               (PRICER_H_W_RATIO_MIN <= h/w <= PRICER_H_W_RATIO_MAX)):
            if test: print("protportion check failed: %f %f"%(w/h, h/w))
            continue
        
        if (w < img.shape[1] * PRICER_W_MIN_PERCENT or
            h < img.shape[0]*PRICER_H_MIN_PERCENT):
            if test: print("size check failed")
            continue
        
        rect_masked = apply_rect_mask(gray, x1, y1, w, h)
        if not hist_filter(rect_masked, (w*h)*HIST_BIN_MIN_RELATIVE,
                           test=test):
            continue
        
        
        barcode = detect_barcode(gray)
        barcode_box = cv2.boundingRect(barcode)        
        if BARCODE_W_H_MIN <= barcode_box[2]/barcode_box[3] <= BARCODE_W_H_MAX:
            x1 = min(x1, barcode_box[0])
            y1 = min(y1, barcode_box[1])
            if(x1 + w < barcode_box[0] + barcode_box[2]):
                w = barcode_box[0] + barcode_box[2] - x1
            if(y1 + h < barcode_box[1] + barcode_box[3]):
                h = barcode_box[1] + barcode_box[3] - y1
        else:
            if test: print("barcode w_h check failed: %f"%
                           (barcode_box[2]/barcode_box[3]))
            
        passed = np.vstack((passed, (x1*scale[1],
                                     (y1 + offset)*scale[0], 
                                     w*scale[1],
                                     h*scale[0])))
    return passed
    

if __name__ == '__main__':
    img = cv2.imread("/home/chernov/google-drive/data/marked_pricers/7.jpg")#"test/price.jpg")
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
    rects = detect_pricer_test(img, test=True)
    for rect in rects:
        cv2.rectangle(img, 
                      (int(rect[0]),int(rect[1])), 
                      (int(rect[0] + rect[2]), 
                       int(rect[1] + rect[3])),
                      (255, 0, 0), 3)
    cv2.imshow("pricer", img)
    cv2.waitKey()
    cv2.destroyAllWindows()