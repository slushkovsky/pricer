#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 23:54:39 2016

@author: chernov
"""

from os import environ, listdir, path, system
import math
import copy
import sys

import cv2
import numpy as np

main_dir = path.dirname(path.dirname(__file__))
if not main_dir in sys.path:
    sys.path.append(main_dir)

from marking_tools.os_utils import show_hist


DATASET_NAME = "names_lines"
DATA_PATH = environ["BEORGDATAGEN"] + "/CData_full"
DATASET_PATH = DATA_PATH + "/" + DATASET_NAME
CLASSIFIER_NM1_PATH = environ["BEORGDATA"] + "/cv_trained_classifiers/trained_classifierNM1.xml"
CLASSIFIER_NM2_PATH = environ["BEORGDATA"] + "/cv_trained_classifiers/trained_classifierNM2.xml"

def calc_variance(image, percent=0.1):
    range_ = (20.0, 255.0)
    hist, bins = np.histogram(image.ravel(), 100, range_)
    hist = hist/hist.sum()
    hist_int = hist.cumsum(axis=0) 
    hist_int = hist_int

    start_after = bins[np.argmax(hist_int > (1.0 - percent))]
    hist, bins = np.histogram(image.ravel(), 100, (int(start_after), 255))
    hist = hist/hist.sum()
    
    return hist.var()
    
    
def adapt_tresh_mask(mask):
    th_window_size =  mask.shape[0]//3
    if not th_window_size % 2:
        th_window_size += 1
    mask_out = cv2.adaptiveThreshold(mask,255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, th_window_size, 2)
    return mask_out
    
    
def find_contours(img):
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    contours_out = []
    for i in range(0, len(contours)):
        if hierarchy[0][i][3] == -1:
            contours_out.append(contours[i])
            
    #гистограмма по ширинам символов
    hist_w = np.zeros(len(contours_out))
    for i in range(0, len(contours_out)):
        x, y, hist_w[i], h = cv2.boundingRect(contours_out[i])
    hist, bins = np.histogram(hist_w, 15)
    hist = hist/hist.sum()
    
    
    hist_integr = np.cumsum(hist, axis=0)
    max_w = bins[np.argmax(hist_integr > 0.92)]
    print(max_w)
    
    contours_big_w = []
    for i in range(0, len(contours_out)):
        if hist_w[i] > max_w:
            contours_big_w.append(contours_out[i])

    
def lines_mask_circle(sobel, step, line_thickness, line_width, 
                      min_intensity=20, min_perc=0.4):
    deg_step = math.pi/180.
    full_mask = np.zeros(sobel.shape, np.uint8)
    for y in range(0, sobel.shape[0], step):
        for x in range(0, sobel.shape[1], step):
            for degree in range(-90, 90, 10):
                x2 = int(x + line_width * math.cos(deg_step*degree))
                y2 = int(y + line_width * math.sin(deg_step*degree))
                mask = np.zeros(sobel.shape, np.uint8)
                cv2.line(mask, (x, y), (x2, y2), 255, line_thickness)
                masked = cv2.bitwise_and(sobel, sobel, mask=mask)
                if(not (masked > min_intensity).sum() > 
                   line_width * line_thickness * min_perc):
                    full_mask = cv2.bitwise_or(full_mask, mask)       
    full_mask = 255 - full_mask
    return full_mask
    
    
def detect_text_cv(img, test=False):
    MAX_W_H_RATIO = 1.5
    IMG_H = 100
    MERGE_STEP_PIXELS = 20
    w = int(IMG_H*img.shape[1]/img.shape[0])
    img = cv2.resize(img, (w, IMG_H))
    
    channels = cv2.text.computeNMChannels(img)

    rects = []
    for channel in channels:
        erc1 = cv2.text.loadClassifierNM1(CLASSIFIER_NM1_PATH)
        er1 = cv2.text.createERFilterNM1(erc1,
                                         16,
                                         0.001,
                                         0.99,
                                         0.2,
                                         False,
                                         0.9)
        erc2 = cv2.text.loadClassifierNM2(CLASSIFIER_NM2_PATH)
        er2 = cv2.text.createERFilterNM2(erc2,0.95)
        regions = cv2.text.detectRegions(channel,er1, er2)
        for i in range(len(regions)):
            x,y,w,h = cv2.boundingRect(regions[i])
            rects.append([x,y,w,h])
    
    rects_merged = []
    if len(rects) > 0:
        rects_merged = [rects[0]]
        for i in range(len(rects)):
            rect = rects[i]
            any_ = False
            for j in range(len(rects_merged)):
                if abs(rect[0] - rects_merged[j][0]) < MERGE_STEP_PIXELS and \
                   abs(rect[1] - rects_merged[j][1]) < MERGE_STEP_PIXELS and \
                   abs(rect[2] - rects_merged[j][2]) < MERGE_STEP_PIXELS and \
                   abs(rect[3] - rects_merged[j][3]) < MERGE_STEP_PIXELS:
                       any_=True
                       rects_merged[j][0] = min(rect[0], rects_merged[j][0])
                       rects_merged[j][1] = min(rect[1], rects_merged[j][1])
                       rects_merged[j][2] = max(rect[2], rects_merged[j][2])
                       rects_merged[j][3] = max(rect[3], rects_merged[j][3])
                       break
            if not any_:
                rects_merged.append(rect)
        
        rects_buf = []
        for i in range(len(rects_merged)):
            rect = rects_merged[i]
            if not rect[2]/rect[3] > MAX_W_H_RATIO:
                rects_buf.append(rect)
        rects_merged = rects_buf
        
        hw_ratios = np.zeros(len(rects_merged))
        widths = np.zeros(len(rects_merged))
        heights = np.zeros(len(rects_merged))
        for i in range(len(rects_merged)):
            rect = rects_merged[i]
            hw_ratios[i] = rect[3]/rect[2]
            widths[i] = rect[2]
            heights[i] = rect[3]

        hist, bins = np.histogram(heights*widths, 20)
        show_hist(hist, bins)
        
        if test:  
            hist, bins = np.histogram(hw_ratios, 20)
            hist1, bins1 = np.histogram(widths, 20)
            hist2, bins2 = np.histogram(heights, 20)
            show_hist(hist, bins)
            show_hist(hist1, bins1)
            show_hist(hist2, bins2)
    
    vis = copy.copy(img)        
    for rect in rects_merged:
        cv2.rectangle(vis, 
                      (rect[0], rect[1]),
                      (rect[0] + rect[2],
                       rect[1] + rect[3]),
                      (0, 0, 255),
                      1)
    cv2.imshow("vis", vis)
    
    
def process_image(img, test=False):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    hist, bins = np.histogram(gray.ravel(), 60)
    hist = hist/hist.sum()
    show_hist(hist, bins)
    
    hist_int = hist.cumsum(axis=0)
    tresh = bins[np.argmax(hist_int > 0.2)]
    print(tresh)
    
    ret, th = cv2.threshold(gray,tresh,255,cv2.THRESH_BINARY_INV)
    ret2,otsu = cv2.threshold(gray, 0, 255,cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    cv2.imshow("tresh", otsu)
    
    print((otsu > 0).sum() / otsu.size)
    
    sobelx = np.abs(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5))
    sobelx = ((sobelx/sobelx.max())*255).astype(np.uint8)
    sobely = np.abs(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5))
    sobely = ((sobely/sobely.max())*255).astype(np.uint8)
    calc_variance(sobelx)
    calc_variance(sobely)
    sobel = sobelx + sobely
    
    cv2.imshow("sobel", sobel)
    im2, contours, hierarchy = cv2.findContours(otsu, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    
    contours_out = []
    for i in range(0, len(contours)):
        if hierarchy[0][i][3] == -1:
            contours_out.append(contours[i])
            
    for contour in contours_out:
        x, y, w, h = cv2.boundingRect(contour)
        symbol = img[y:y+h, x:x+w]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255),2)
        
    cv2.imshow("B", img)


def mask_full_lines(img, test=False):
    LINE_THICKNESS = 1
    STEP = 1
    LINE_WIDTH = 5
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("gray" , gray)
    
    sobelx = np.abs(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3))
    sobelx = ((sobelx/sobelx.max())*255).astype(np.uint8)
    sobely = np.abs(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3))
    sobely = ((sobely/sobely.max())*255).astype(np.uint8)
    calc_variance(sobelx)
    calc_variance(sobely)
    sobel = sobelx + sobely
    
    hist, bins = np.histogram(sobel.ravel(), 40)
    hist = hist/hist.sum()
    show_hist(hist, bins)
    
    min_int = bins[np.argmax(hist.cumsum(axis=0) > (1.0 - 0.15))]
    print(min_int)
    ret,mask = cv2.threshold(sobel,min_int,255,cv2.THRESH_BINARY)
    sobel = cv2.resize(sobel, (600, 300), cv2.INTER_CUBIC)
    cv2.imshow("sobel" , sobel)
    
    full_mask = lines_mask_circle(sobel, STEP, LINE_THICKNESS, LINE_WIDTH)
    if test:
        cv2.imshow("mask_full_hor_lines mask", full_mask)  
    return full_mask

    
if __name__ == "__main__":
    if not path.exists(DATA_PATH):
        print("%s not found, start croping"%(DATA_PATH))
        system("python3 ../marking_tools/crop_pricers.py")
        print("croping done")

    if not path.exists(DATASET_PATH):
        print("%s not found, start croping"%(DATASET_PATH))
        system("python3 ../marking_tools/crop_pricer_fields.py")
        print("croping done")
    else:
        print("%s found, skip croping. To force restart delete this folder"
              %(DATASET_PATH))
    
    i = 0
    for image_name in listdir(DATASET_PATH):
        if i < 0:
            i += 1
            continue
        
        image_path = path.join(DATASET_PATH, image_name)
        print(image_path)
        img = cv2.imread(image_path)
        
        #img = cv2.bilateralFilter(img,12, 100, 75)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        
        for i in range(img.shape[2]):
           img[:,:,i] = clahe.apply(img[:,:,i])
        
        #kernel = np.ones((5,10),np.uint8)
        #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        #cv2.imshow("clahe", img)
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        mask = detect_text_cv(img)
        k = cv2.waitKey() & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
