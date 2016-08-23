#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 23:51:24 2016

@author: chernov
"""

from argparse import ArgumentParser

import cv2
import numpy as np

def process_image(img, min_h_mask=0.15, min_symb_w=0.01,
                  min_symb_h=0.4, thresh_steps=6, min_variance=1000):
    """
      Выделение символов на ценовых полях ценника.
      
      Алгоритм:
          1. Разделение изображения на thresh_steps частей.
          2. Бинаризация методом Отсу каждой части.
          3. Построение гистограммы путем суммирования пикслелей в столбцах.
             Нормализация гистограммы.
          4. Фильтрация гистограммы по пороговому значению min_h_mask.
          5. Нахождение контуров.
          6. Фильтрация контуров по вариации внутри контура и размерам.
      
      @img - Входное изображение.
      @min_h_mask - Пороговое значение для фильтрации гистограммы
      @min_symb_w - минимальная ширина символа в относительном размере
      @min_symb_h - минимальная высота символа в относительном размере
      @thresh_steps - число блоков для бинаризации
      @min_variance - минимальная вариация в контуре символа
      
      @rtype list(list())
    """
    
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    step = img.shape[1]//thresh_steps
    otsu = np.zeros(gray.shape, np.uint8)
    for i in range(0, img.shape[1], step):
        ret2, otsu[:, i:i+step] = cv2.threshold(gray[:, i:i+step], 
                                                0, 255, cv2.THRESH_OTSU +
                                                cv2.THRESH_BINARY_INV)
    
    otsu_hist = otsu.sum(axis=0)
    otsu_hist = otsu_hist/otsu_hist.max()
    otsu[:, otsu_hist < min_h_mask ] = 0 

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
                min(gray.shape[1], x + int(w*6/5))].var() < min_variance:
            rects_bad.append([x,y,w,h])
            continue
        
        if (w > img.shape[1]*min_symb_w and
            h > img.shape[0]*min_symb_h):
            rects.append([x, y, w, h])
        else:
            rects_bad.append([x,y,w,h])
            
    rects = np.array(rects)
    if len(rects):
        rects[:, 1] = rects[:, 1].min()
        rects[:, 3] = rects[:, 3].max()
        rects = tuple(rects)  
    return rects


def parse_args():   
    parser = ArgumentParser()
    parser.add_argument('image', type=str,
                        help="Image filepath")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
        
    img = cv2.imread(args.image)
    rects =  process_image(img)
    
    vis = img.copy()
    for rect in rects:
        cv2.rectangle(vis, (rect[0], rect[1]),
                      (rect[0] + rect[2], rect[1] + rect[3]),
                      (0, 0, 255), 2)

    cv2.imshow("vis", vis)
    cv2.waitKey()
    cv2.destroyAllWindows()