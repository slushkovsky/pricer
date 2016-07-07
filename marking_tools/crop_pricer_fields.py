#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 23:22:23 2016

@author: chernov
"""

import shutil
from os import path, makedirs, environ

import cv2
import numpy as np

from markers_db_tools import get_symbol_image

MARK_DATA_PATH = environ["BEORGDATA"] + "/marked_pricers/local_data.txt"
DATA_PATH = environ["BEORGDATA"] + "/marked_pricers"
OUT_PATH = environ["BEORGDATAGEN"] + "/CData_full"

RUBLI_LOCALISATION_PATH =  DATA_PATH + "/rubli.txt"
KOPEIKI_LOCALISATION_PATH =  DATA_PATH + "/kopeiki.txt"
NAMES_LOCALISATION_PATH =  DATA_PATH + "/name.txt"

def generate_crop_data(data_path, out_name, orig_size=(500,250)):
    out_path = "%s/%s"%(OUT_PATH, out_name)
    
    if(path.exists(out_path)):
        shutil.rmtree(out_path)
    makedirs(out_path)
    
    with open(data_path, 'r') as mark_file:
        for line in mark_file.readlines():
            s = line.split()
            img =  cv2.imread(OUT_PATH + "/" + s[0])
            rows, cols, ch  = img.shape
            
            cnt = np.zeros(8, np.float)
            for i in range(1, 9, 2):
                cnt[i - 1] = float(s[i]) / orig_size[0] * cols
                cnt[i] = float(s[i + 1]) / orig_size[1] * rows

            cnt = cnt.reshape((4,2)).astype(np.int)

            x,y,w,h = cv2.boundingRect(cnt)
            img = img[y :y + h, x:x + w]
            cv2.imwrite("%s/%s"%(out_path, s[0]), img)
            
            
generate_crop_data(RUBLI_LOCALISATION_PATH, "rubli")
generate_crop_data(KOPEIKI_LOCALISATION_PATH, "kopeiki")
generate_crop_data(NAMES_LOCALISATION_PATH, "names")