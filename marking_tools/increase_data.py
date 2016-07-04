#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 18:51:07 2016

@author: Artem Lukoyanov, chernov
"""

import random
import copy

import cv2
import numpy as np

def rotate_image(image, angle):
  image_center = (image.shape[0] / 2, image.shape[1] / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  shape = (image.shape[0], image.shape[1])
  result = cv2.warpAffine(image, rot_mat, shape,flags=cv2.INTER_LINEAR)
  return result


def increase_data(image, iterations=5):
    out = []
    for i in range(iterations):
        img = copy.copy(image)
        rows,cols, ch  = img.shape
        
        ###########BLURING
        ker1 = int (random.random() * 15) + 1
        ker2 = int (random.random() * 15) + 1
        img = cv2.blur(img,(ker1, ker2))
        ##################
        
        ############COLORS
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = hsv[:,:,0] + int(40 * (random.random() - 0.5))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        ##################
        
        #############GAMMA
        gamma = random.random() * 2.5 + 0.25
        invGamma = 1.0 / gamma
        table = np.array([((j / 255.0) ** invGamma) * 255
        	for j in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)
        ##################
        
        # Из-за поворота на изображении появляется черный задний фон
        #degrees = random.uniform(-5,5)
        #img = rotate_image(img, degrees)
        out.append(img)
    return out

