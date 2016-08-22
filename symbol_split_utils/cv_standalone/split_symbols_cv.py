#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 23:54:39 2016

@author: chernov
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import interpolate

__DEBUG__ = False
    
def metrical_interp(img, samples, knn_step=25, min_nn=5):
    """Метрическая интерполяция 2-мерной фунции
    
    Алогритм
        - Построение двумерной сетки с шагом knn_step
        - Аппроксимация значений функции в каждом узле сетки с помощью knn.
          Точка в узле вычисляется как линейная комбинация n ближайших точек,
          входящих в ЛК пропорционально их расстоянию до узла
        - Интерполяция полученной сетки в непрерывную фунцию с помощью сплайна
    
    Keyword arguments:
        img -- входное изображение
        samples -- Точки для аппроксимации в формате [[y, x, value],...]
        knn_step -- щаг сетки knn
        min_nn -- минимальное количество ближайших точек для аппроксимации
    
    """
    img = img.copy()
    
    knn = cv2.ml.KNearest_create()
    knn.train(samples[:, 0:2], 0, samples[:, 2])
    
    x = np.arange(0, img.shape[1], knn_step)
    y = np.arange(0, img.shape[0], knn_step)
    k = max(min(min_nn, samples.shape[0]), samples.shape[0]//3)
    
    samples_grid = np.zeros((len(y), len(x)), np.float32)
    for j, y_cur in enumerate(y):
        for i, x_cur in enumerate(x):
            point = np.array([[y_cur, x_cur]],np.float32)
            ret, results, neighbours, dist = knn.findNearest(point, k)
            coeffs = 1 - (dist / dist.sum())**2.0
            coeffs = coeffs / coeffs.sum()
            val = (neighbours*coeffs).sum()
            samples_grid[j, i] = val 

    if __DEBUG__:
        plt.imshow(samples_grid, interpolation='nearest', cmap=cm.gist_rainbow)
        plt.show()
        
    ky = min(3, len(y) - 1)
    kx = min(3, len(x) - 1)
    f = interpolate.RectBivariateSpline(y, x, samples_grid, kx=ky, ky=kx)
    return f  
   
    
def detect_MSER_boxes(img, min_variance, max_wh_ratio, min_symb_h,
                      merge_step_pixels):
    """Деткетирование и фильтрация MSER боксов

    Боксы MSER регионов фильтруются по вариации внутри бокса, отношению ширины
    к высоте и минимальной высоте символа
    
    Keyword arguments:
        img -- входное изображение
        min_variance -- минимальная вариация в боксе символа
        max_wh_ratio -- максимальное отношение ширины к высоте бокса
        min_symb_h -- минимальная высота символа в пикселях
        merge_step_pixels -- шаг слияния боксов в символах
    """
    
    channels = cv2.text.computeNMChannels(img)
    
    rects = []
    for channel in channels:   
        mser = cv2.MSER_create()
        regions = mser.detectRegions(channel)[0]
        for i in range(len(regions)):
            x,y,w,h = cv2.boundingRect(regions[i])
            rects.append([x,y,w,h])
    
    rects_merged = []
    rects_outside = []
    rects_inside = []
    if len(rects) > 0:
        rects_merged = merge_close_rects(rects, merge_step_pixels)
        rects_merged = filter_rects_wh(rects_merged, max_wh_ratio,
                                       min_symb_h)        
        rects_outside, rects_inside = find_rects_inside(rects_merged)
        rects_outside = filter_rects_variance(img, rects_outside, min_variance)
    return rects_outside, rects_inside
    

def normalize_crop(img, mean_in_rect=160, min_variance=200, max_wh_ratio=1.5,
                   min_symb_h=0, merge_step_pixels=20):
    """Нормализция изображения с помощью метрического анализа
    
    Алгоритм:
        - Вычисление MSER боксов, в которых потенциально есть символы
        - Вычисление коэффициентов нормализации яркости в боксах
        - Интерполяция коэффициэентов на все изображение с помощью метрического
        анализа
        - Применение нормализации на полном изображении
        
    Keyword arguments:
        img -- входное изображение
        mean_in_rect -- среднее значение яркости в нормализованном боксе символа
        min_variance -- минимальная вариация в боксе символа
        max_wh_ratio -- максимальное отношение ширины к высоте бокса
        min_symb_h -- минимальная высота символа в пикселях
        merge_step_pixels -- шаг слияния боксов в символах
        
    """
    img = img.copy()
    rects_outside,\
    rects_inside = detect_MSER_boxes(img,
                                     min_variance=min_variance,
                                     max_wh_ratio=max_wh_ratio,
                                     min_symb_h=min_symb_h,
                                     merge_step_pixels=merge_step_pixels)
    
    samples = np.zeros((len(rects_outside), 3), np.float32)    
    for i, rect in enumerate(rects_outside):
        samples[i, 0] = rect[0] + rect[2]/2
        samples[i, 1] = rect[1] + rect[3]/2

        box = img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[0]]
        samples[i, 2] = mean_in_rect / box.mean()
    
    f = metrical_interp(img, samples)
    
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    z = f(y, x)
    
    if __DEBUG__:
        plt.imshow(z, interpolation='nearest', cmap=cm.gist_rainbow)
        plt.show()
        plt.imshow(img)
        plt.show()
    
    for i in range(img.shape[2]):   
        normalized = img[:, :, i] * z
        normalized[normalized > 255.0] = 255.0
        img[:, :, i] = (normalized).astype(np.uint8)
    
    if __DEBUG__:
        plt.imshow(img)
        plt.show()
        
    return img

    
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


def filter_rects_wh(rects, max_wh_ratio=1.5, min_h=None):
    rects_buf = []

    for i in range(len(rects)):
        rect = rects[i]
        if min_h and rect[3] < min_h:
            continue
        
        if not rect[2]/rect[3] > max_wh_ratio:
            rects_buf.append(rect)
            
    return rects_buf
    
def filter_rects_variance(img, rects, min_var):
    rects_filtered = []
    for rect in rects:
        symb = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0] + rect[2]]
        if symb.var() > min_var:
            rects_filtered.append(rect)
        
        #plt.imshow(symb)
        #plt.show()
        #print(symb.var())
    return rects_filtered


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
    
    
def resize_h(img, new_h):
    w = int(new_h*img.shape[1]/img.shape[0])
    scale = np.array(img.shape)
    img = cv2.resize(img, (w, new_h))
    scale = scale/np.array(img.shape)
    return img, scale
    
    
def detect_text_cv(img, nm1_path, nm2_path, min_variance=200,
                   max_wh_ratio=1.5, img_h=300, min_symb_h=0,
                   merge_step_pixels=20):
    """ Детектирование символов на изображении с помощью текстового
    модуля opencv.
    
    Keyword arguments:
        img -- Входное цветное изображенияю
        nm1_path -- Путь к файлу обученного NM1 классификатора.
        nm2_path -- Путь к файлу обученного NM2 классификатора.
        min_variance -- Минимальная вариация внутри контура с символом.
        Чем меньше значение параметра, тем более смазанные изображения будут
        проходить через фильтрю
        max_wh_ratio -- Максимальное соотношение ширины к высоте символа.
        img_h -- Высота в пикселях, к которой будет пережато изображение.
        min_symb_h -- Минимальная высота символа в пикселях.
        merge_step_pixels -- Минимальное смещение между отдельными контурами.
        Контуры, смещенные меньше этого параметра будут объеденены.
        
    """
    img, scale = resize_h(img, img_h)
    
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(img_h//3,img_h//3))
    #for i in range(img.shape[2]):
    #    img[:,:,i] = clahe.apply(img[:,:,i])
    
    channels = cv2.text.computeNMChannels(img)
   
    vis = img.copy()

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
        if False:
            rects_ = cv2.text.erGrouping(img, channel,
                                         [r.tolist() for r in regions])
                                                        
            for r in range(0,np.shape(rects_)[0]):
                rect = rects_[r]
                cv2.rectangle(vis, (rect[0],rect[1]),
                              (rect[0]+rect[2],rect[1]+rect[3]),
                              (0, 255, 255), 2)
      
        for i in range(len(regions)):
            x,y,w,h = cv2.boundingRect(regions[i])
            rects.append([x,y,w,h])
    
    rects_merged = []
    rects_outside = []
    rects_inside = []
    if len(rects) > 0:
        rects_merged = merge_close_rects(rects, merge_step_pixels)
        rects_merged = filter_rects_wh(rects_merged, max_wh_ratio,
                                       min_symb_h)        
        rects_outside, rects_inside = find_rects_inside(rects_merged)
        rects_outside = filter_rects_variance(img, rects_outside, min_variance)
      
    rects_outside = rescale_rects(rects_outside, scale)
    rects_inside = rescale_rects(rects_inside, scale)    
    return rects_outside, rects_inside
