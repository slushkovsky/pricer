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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import cv2

from marking_tools.os_utils import show_hist
from utils.filters import monochrome_mask
from otsu import otsu

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
    
    
def extract_gradients(gray, mask):
    binary = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    binary = ((binary / binary.max())*255).astype(np.uint8)
    
    gray_mask = cv2.bitwise_and(gray, gray, mask=mask)
    
    ret, mask = cv2.threshold(gray_mask, 1,255, 
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, binary, mask=mask)
    
    hist, bins = np.histogram(binary.ravel(), 255,  (1, 255))
    #hist = hist.cumsum()
    #hist = hist - hist.min()
    #hist = hist/hist.max()
    #hist = hist/float(binary.max())*255.0
    show_hist(hist, bins)
        
    #for i in range(binary.shape[1]):
    #    for j in range(binary.shape[0]):
    #        binary[i,j] *= hist[binary[i,j]]
    
    #binary = binary.astype(np.uint8)
    
    cv2.imshow("binary", binary)
    cv2.waitKey()
    
    return binary
    
    
def otsu_iterations(color, noise_h=1.0, symbol_max_h=0.9):
    gray = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV_FULL)
    v = hsv[:,:, 2]

    v_mask_dark = np.full(gray.shape, 255, np.uint8)
    v_mask_dark_full = np.full(gray.shape, 255, np.uint8)
    v_ = v.copy()
    t = 0
    
    for i in range(5):
        t_cur = otsu(v_, True)
        if t == t_cur:
            break
        t = t_cur
        
        ret, v_mask_cur = cv2.threshold(v_, t, 255, cv2.THRESH_BINARY)
        dark_cols = 1.0 - (v_mask_cur.sum(axis=0).astype(np.float)/255/img.shape[0])
        
        v_mask_cur[:, dark_cols >= symbol_max_h] = 255
        v_mask_dark = cv2.bitwise_and(v_mask_dark, v_mask_cur)

        cur_ful_mask = np.full(gray.shape, 255, np.uint8)
        cur_ful_mask[:, dark_cols >= noise_h] = 0
        cur_ful_mask[:, v_mask_dark_full[0] == 0] = 255

        v_mask_dark_full[:, dark_cols >= noise_h] = 0
        v_ = cv2.bitwise_and(v, v, mask=255 - v_mask_dark_full)

        coverage = cur_ful_mask[:, dark_cols > 0.9].shape[1]/img.shape[1]

        #cv2.imshow("cur_ful_mask",cur_ful_mask)
        #cv2.imshow("v_mask_dark_full", v_mask_dark_full)
        #cv2.imshow("v_mask_dark", v_mask_dark)
        #cv2.imshow("v_", v_)
        #cv2.waitKey()

        if coverage > 0.1:
            v_mask_dark_full[:, dark_cols > 0.9] = 0
        else:
            break
        
    v_mask = np.full(gray.shape, 255, np.uint8)
    v_mask_full = np.full(gray.shape, 255, np.uint8)
    t = 0
    for i in range(2):
        v_ = cv2.bitwise_and(v, v, mask=v_mask_cur)
        
        t_cur = otsu(v_, True)
        
        if t_cur == t:
            break
        
        t = t_cur
        ret, v_mask_cur = cv2.threshold(v_, t, 255, cv2.THRESH_BINARY)
        
        dark_cols = 1.0 - (v_mask_cur.sum(axis=0).astype(np.float)/255/img.shape[0])
        
        v_mask_cur[:, dark_cols > symbol_max_h] = 255
        v_mask_full[:, np.logical_and(dark_cols <= symbol_max_h,
                                      dark_cols > 0.05)] = 0
        v_mask = cv2.bitwise_and(v_mask, v_mask_cur)
        
        #cv2.imshow("v_mask_full",v_mask_full)
        #cv2.imshow("v_mask_cur", v_mask_cur)
        #cv2.imshow("v_", v_)
        #cv2.waitKey()
        
    cv2.imshow("v_mask_dark", v_mask_dark)
    cv2.imshow("v_mask", v_mask)
    cv2.waitKey()
    return v_mask, v_mask_dark
  
    
def get_begin_end_text_line(img, h_crop_perc=0.0,
                            treshold_coeff=0.3,
                            conv_size = 0.1,
                            test=False):
    BIG_MAXIMA_ORDER = 0.05
    
    img_h = img.shape[0]
    line_croped = img[int(img_h*h_crop_perc):int(img_h*(1-h_crop_perc)),:]
    
    gray = cv2.cvtColor(line_croped,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    
    hsv = cv2.cvtColor(line_croped, cv2.COLOR_BGR2HSV_FULL)
    v = hsv[:,:, 2]

    steps = 40
    step = gray.shape[1] // steps
    mask = np.full(gray.shape, 255, np.uint8)
    mask_single = np.full(gray.shape, 255, np.uint8)
    mask_two = np.full(gray.shape, 255, np.uint8)
    
    x_val = np.zeros(steps + 2)
    y_val = np.zeros(steps + 2)
    
    for i in range(steps):
        x = step*i
        
        part = gray[:, x:x+step]
        
        t = otsu(part)
        ret, mask_1 = cv2.threshold(part, t, 255, cv2.THRESH_BINARY_INV)
        
        part_2 = cv2.bitwise_and(part, part, mask=mask_1)
        
        t2 = otsu(part_2, True)
        mask_2 = np.logical_or(part_2 != 0, part_2 < t2).astype(np.uint8) * 255
        
        print(t, t2)
        
        mask_ = (np.logical_and(part < t, part < t2)).astype(np.uint8) * 255
        mask[:, x:x+step] = mask_
        mask_single[:, x:x+step] = (part > t).astype(np.uint8) * 255
        mask_two[:, x:x+step] = mask_2
 
        x_val[i + 1] = x + step/2
        y_val[i + 1] = t
        
    
    x_val[0] = 0
    y_val[0] = y_val[1]
    x_val[-1] = gray.shape[1]
    y_val[-1] = y_val[-2]

    f = interp1d(x_val, y_val, kind='slinear')
    x_new = np.arange(gray.shape[1])
    
    plt.plot(x_val, y_val, 'o', x_new, f(x_new), '-')
    plt.show()
        
    cv2.imshow("mask", mask)
    cv2.imshow("mask_single", mask_single)
    cv2.imshow("mask_two", mask_two)
    cv2.waitKey()    
    
    return []
    #kernel = np.ones((gray.shape[0] // 10, gray.shape[0] // 10),np.uint8)
    #v_mask = cv2.dilate(v_mask, kernel, iterations = 1)
    
    v_mask_inv = 255 - v_mask
    
    cv2.imshow("v_mask", v_mask)
    cv2.imshow("v_mask_inv", v_mask_inv)
    
    binary_spec = extract_gradients(gray, v_mask)
    binary_non_spec = extract_gradients(gray, v_mask_inv)
    
    binary = cv2.bitwise_or(binary_spec, binary_non_spec)
    
    #cv2.imshow("binary", binary)
    #cv2.waitKey()

    hist = binary.sum(axis=0)
    #show_hist(hist, np.arange(len(hist) + 1))
    return extract_regions_from_hist(hist, hist.mean()*1.5)
    
    return []
    
    hist = hist/hist.sum()
    var_hist = np.array(())
    
    step = (hist.max() - hist.min()) / 100
    for i in np.arange(hist.min(), hist.max(), step):
        var_hist = np.append(var_hist, hist[hist > i].std())
        
    show_hist(var_hist, np.arange(len(var_hist) + 1))
    return []
    
    hist_3 = np.convolve(hist, np.ones(len(hist)//50), 'same')
    show_hist(hist_3, np.arange(len(hist_3) + 1))
    

    minima = argrelextrema(hist_3, np.less, order=int(len(hist_3)*BIG_MAXIMA_ORDER))[0]
    print(hist_3[minima])
    
    max_var = 0
    min_ = minima[0]
    for cur_minima in minima:
        step = len(hist_3) // 50
        var = hist_3[cur_minima - step: cur_minima + step].var()
        if var > max_var:
            max_var = var
            min_ = cur_minima
    
    min_ = hist_3[min_]
    print(min_)
    return extract_regions_from_hist(hist_3, min_)
    

    hist_2, bins = np.histogram(hist, 200)
    
    hist_2 = np.convolve(hist_2, np.ones(40), 'same')
    
    if test:
        show_hist(hist_2, bins)
        
    # нахождение последнего большого локального максимума
    # соответсвующего тексту
    maxima = argrelextrema(hist_2, np.greater,
                           order=int(len(hist_2)*BIG_MAXIMA_ORDER))[0]
    print(bins[maxima])
    return []                       
                           
    minima = argrelextrema(hist_2, np.less, order=1)[0]
   

    sum_hist = np.cumsum(hist_2/hist_2.sum())
    
    minima = np.delete(minima, np.where(bins[minima] > bins.max()*0.4))

    if test:
        print("minima", bins[minima])
        print("maxima", bins[maxima])
        #show_hist(sum_hist, bins)
        
    
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
    