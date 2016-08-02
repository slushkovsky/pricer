#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 22:18:37 2016

@author: chernov
"""

import numpy as np
from scipy.signal import argrelextrema
import cv2

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
    
    conv_kern_size = int(len(hist) * conv_kern_size)
    conv_kern = np.ones(conv_kern_size)
    hist_sm = np.convolve(hist, conv_kern, 'same')
    hist_sm = hist_sm/hist_sm.max()

    
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
