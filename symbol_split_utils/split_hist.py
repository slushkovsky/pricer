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
    """Выделение регионов по гистограмме и порогу
    
    Keyword arguments:
    hist -- Входная гистограмма.
    treshold -- Порог. Может быть или числом или массивом с числами такой же
                длины как и гистограмма.
    """
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
    
    
def crop_regions(parts, minima_merge_tresh = 10):
    """Удаление мелких регионов
    
    Keyword arguments:
    parts -- регионы
    minima_merge_tresh -- минимальный размер региона в пикселях
    """
    parts_merged = []
    for part in parts:
        if abs(part[1] - part[0]) > minima_merge_tresh:
            parts_merged.append(part)
    return parts_merged
  
    
def get_begin_end_text_line(img, h_crop_perc=0.3,
                            treshold_coeff=0.3,
                            conv_size = 0.1,
                            test=False):
    img_h = img.shape[0]
    line_croped = img[int(img_h*h_crop_perc):int(img_h*(1-h_crop_perc)),:]

    gray = cv2.cvtColor(line_croped,cv2.COLOR_BGR2GRAY)
    binary = np.abs(cv2.Laplacian(gray, cv2.CV_64F))

    hist = binary.sum(axis=0)

    hist_2, bins = np.histogram(hist, 200)
    hist_2 = np.convolve(hist_2, np.ones(20), 'same')
    
    minima = argrelextrema(hist_2, np.less, order=1)[0]
    maxima = argrelextrema(hist_2, np.greater, order=len(hist_2)//4)[0]

    sum_hist = np.cumsum(hist_2/hist_2.sum())
    
    minima = np.delete(minima, np.where(bins[minima] > bins.max()*0.4))

    if test:
        print("minima", bins[minima])
        print("maxima", bins[maxima])
        #show_hist(sum_hist, bins)
        show_hist(hist_2, bins)
    
    treshold = 0
    print(len(maxima))
    if len(maxima) >= 2:
        min_inside = np.where(np.logical_and(minima >= maxima[0],
                                             minima <= maxima[1]))[0]
        if len(min_inside) > 0:
            treshold = bins[minima[min_inside.min()]]
            print(sum_hist[minima[min_inside.min()]])
            
    if test:
        print(treshold)
            
    return extract_regions_from_hist(hist, treshold)
    
    
def split_lines_hist(img, minima_coef=1.001, minima_merge_tresh=0.15,
                     conv_kern_size=0.05, extremum_order=0.125,
                     test=False):
    """Разделение текста на строки по построчной гистограмме суммы лапласианов.
    
    Описание алгоритма:
    1. К изображению применяется лаплассиан
    2. На лапласиане для каждой строки ищется сумма интенсивностей. Таким 
      образом строится построчная гистограмма
    3. На полученной гистограмме находятся локальные минимумы и максимумы
      - Если между двумя максимумами выполняется соотношение
        var/mean*mean_global/var_global < minima_merge_tresh
        var, mean - вариация и среднее в промежутке между двумя максимумами;
        var_global, mean_global - вариация и среднее всей гистограмме;
        то локальные минимумы в этом промежутке не берутся.
    4. Локальные максимумы дополняются граничными линиями. Для каждого проме-
      жутка между локальными максимумами в качестве порога выбирается значе-
      ние ближайшего минимума, умноженное на коэффициент minima_coef.
    
    Keyword arguments:
    img -- цветное или серое изображение
    minima_coef -- Коэффициент услиления локального минимума.
    conv_kern_size -- Размер светрочного окна (в процентах от высоты
                               изображения) при сглаживании гистограммы.
    minima_merge_tresh -- Порог отсеивания локальных минимумов (см п.3).
    extremum_order -- Размер окна (в процентах от высоты изображения) при 
                      поиске локальных экстремумов
    
    """
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    binary = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    
    hist = binary.sum(axis=1)
    hist = hist/hist.max()
    if test:
        show_hist(hist, np.arange(len(hist) + 1))
    
    conv_kern_size = int(len(hist) * conv_kern_size)
    conv_kern = np.ones(conv_kern_size)
    hist_sm = np.convolve(hist, conv_kern, 'same')
    hist_sm = hist_sm/hist_sm.max()
    
    if test:
        show_hist(hist_sm, np.arange(len(hist_sm) + 1))
    
    order = int(len(hist_sm) * extremum_order)
    
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
        if coef < minima_merge_tresh:
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
            th[maxima[i]: maxima[i + 1]] = hist_sm[th_cur] * minima_coef
    else:
        hist_min = hist_sm.min()
        if hist_min:
            th = hist_sm.min() * minima_coef
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
    