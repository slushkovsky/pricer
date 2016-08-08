#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 23:54:39 2016

@author: chernov
"""

from argparse import ArgumentParser
from os import environ, listdir, path, system, makedirs
import math
import copy
import sys

import cv2
import numpy as np

main_dir = path.dirname(path.dirname(path.abspath(__file__)))
if not main_dir in sys.path:
    sys.path.append(main_dir)

from utils.os_utils import show_hist


DATASET_NAME = "rubli"
DATA_PATH = environ["BEORGDATAGEN"] + "/CData_full"
DATASET_PATH = DATA_PATH + "/" + DATASET_NAME
CLASSIFIER_NM1_PATH = environ["BEORGDATAGEN"] + "/cv_trained_classifiers/trained_classifierNM1.xml"
CLASSIFIER_NM2_PATH = environ["BEORGDATAGEN"] + "/cv_trained_classifiers/trained_classifierNM2.xml"

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
    scale = np.array(img.shape)
    img = cv2.resize(img, (w, IMG_H))
    scale = scale/np.array(img.shape)
    
    channels = cv2.text.computeNMChannels(img)

    rects = []
    for channel in channels:
        erc1 = cv2.text.loadClassifierNM1(CLASSIFIER_NM1_PATH)
        er1 = cv2.text.createERFilterNM1(erc1,
                                         1,
                                         0.05,
                                         0.95,
                                         0.8,
                                         True,
                                         0.4)
        erc2 = cv2.text.loadClassifierNM2(CLASSIFIER_NM2_PATH)
        er2 = cv2.text.createERFilterNM2(erc2,0.5)
        regions = cv2.text.detectRegions(channel,er1, er2)
        for i in range(len(regions)):
            x,y,w,h = cv2.boundingRect(regions[i])
            rects.append([x,y,w,h])
    
    rects_merged = []
    rects_bad = []
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
        
        rects_buf = []
        for i in range(len(rects_merged)):
            contains = False
            for j in range(len(rects_merged)):
                rect_i = rects_merged[i]
                rect_j = rects_merged[j]
                if(rect_i != rect_j and
                   rect_i[0] >= rect_j[0] and
                   rect_i[1] >= rect_j[1] and
                   rect_i[0] - rect_j[0] + rect_i[2] <= rect_j[2] and
                   rect_i[1] - rect_j[1] + rect_i[3] <= rect_j[3]):
                    contains = True
                    break
            if not contains:
                rects_buf.append(rect_i)
            else:
                rects_bad.append(rect_i)
                
        rects_merged = rects_buf
        
        hw_ratios = np.zeros(len(rects_merged))
        widths = np.zeros(len(rects_merged))
        heights = np.zeros(len(rects_merged))
        for i in range(len(rects_merged)):
            rect = rects_merged[i]
            hw_ratios[i] = rect[3]/rect[2]
            widths[i] = rect[2]
            heights[i] = rect[3]
        
        if test:  
            hist, bins = np.histogram(hw_ratios, 20)
            hist1, bins1 = np.histogram(widths, 20)
            hist2, bins2 = np.histogram(heights, 20)
            print("hw")
            show_hist(hist, bins)
            print("w")
            show_hist(hist1, bins1)
            print("h")
            show_hist(hist2, bins2)
            hist, bins = np.histogram(heights*widths, 20)
            print("area")
            show_hist(hist, bins)
            
    for i in range(len(rects_bad)):
        rects_bad[i][0] = int(rects_bad[i][0] * scale[1])
        rects_bad[i][2] = int(rects_bad[i][2] * scale[1])
        rects_bad[i][1] = int(rects_bad[i][1] * scale[0])
        rects_bad[i][3] = int(rects_bad[i][3] * scale[0])
    
    for i in range(len(rects_merged)):
        rects_merged[i][0] = int(rects_merged[i][0] * scale[1])
        rects_merged[i][2] = int(rects_merged[i][2] * scale[1])
        rects_merged[i][1] = int(rects_merged[i][1] * scale[0])
        rects_merged[i][3] = int(rects_merged[i][3] * scale[0])
        
    return rects_merged, rects_bad
    
    
def process_image(img, test=False):
    global idx
    MIN_H_MASK = 0.15
    MIN_SYMB_W = 0.01
    MIN_SYMB_H = 0.4
    TRESH_STEPS = 6
    MIN_VARIANCE = 1000
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #ret2,otsu = cv2.threshold(gray, 0, 255, 
    #                          cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    
    step = img.shape[1]//TRESH_STEPS
    otsu = np.zeros(gray.shape, np.uint8)
    for i in range(0, img.shape[1], step):
        ret2, otsu[:, i:i+step] = cv2.threshold(gray[:, i:i+step], 
                                                0, 255, cv2.THRESH_OTSU +
                                                cv2.THRESH_BINARY_INV)
    
    otsu_hist = otsu.sum(axis=0)
    otsu_hist = otsu_hist/otsu_hist.max()
    
    #show_hist(otsu_hist, np.arange(otsu.shape[1] + 1))
    
    otsu[:, otsu_hist < MIN_H_MASK ] = 0
    cv2.imshow("tresh", otsu)    

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
    #print (np.array(rects).shape)    
    return rects, rects_bad


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
    

def parse_args():   
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help="Folder to save reults")
    parser.add_argument('--save_path', type=str,
                        help="Folder to save reults")
    return parser.parse_args()

 
if __name__ == "__main__":
    
    args = parse_args()
    
    save_path = args.save_path
    
    images = None
    dataset_path = None
    if not args.dataset_path:
        if not path.exists(DATA_PATH):
            print("%s not found, start croping"%(DATA_PATH))
            system("python3 ../marking_tools/crop_pricers.py")
            print("croping done")
    
        names_path = path.join(DATA_PATH, "names")
        if not path.exists(names_path):
            print("%s not found, start croping"%(names_path))
            system("python3 ../marking_tools/crop_pricer_fields.py")
            print("croping done")
        
        if not path.exists(DATASET_PATH):
            print("%s not found, start croping"%(DATASET_PATH))
            system("python3 %s"%(path.join(path.dirname(__file__),
                                           "extract_text_lines.py")))
            print("croping done")
        else:
            print("%s found, skip croping. To force restart delete this folder"
                  %(DATASET_PATH))
        dataset_path = DATASET_PATH
        images = listdir(dataset_path)
    else:
        if path.isdir(args.dataset_path):
            dataset_path = args.dataset_path
            images = listdir(dataset_path)
        else:
            dataset_path = path.abspath(path.dirname(args.dataset_path))
            images = [path.basename(args.dataset_path)]
            
    
    if save_path and not path.exists(save_path):
        makedirs(save_path)
    
    for image_name in images:
        if not image_name.endswith("jpg"):
            continue
        
        image_path = path.join(dataset_path, image_name)
        print(image_path)
        img = cv2.imread(image_path)
        
        #rects, rects_inside = detect_text_cv(img)
        rects, rects_inside =  process_image(img, True)
        
        vis = copy.copy(img)   
        for rect in rects:
            cv2.rectangle(vis, (rect[0], rect[1]),
                          (rect[0] + rect[2], rect[1] + rect[3]),
                          (0, 0, 255), 2)
        for rect in rects_inside:
            cv2.rectangle(vis, (rect[0], rect[1]),
                          (rect[0] + rect[2], rect[1] + rect[3]),
                          (0, 255, 255), 1)
        if save_path:
            cv2.imwrite(path.join(save_path, image_name), vis)
        else:
            cv2.imshow("vis", vis)
            
            k = cv2.waitKey() & 0xFF
            if k == 27:
                break

    cv2.destroyAllWindows()
