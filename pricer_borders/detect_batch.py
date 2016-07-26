#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 23:52:47 2016

@author: chernov
"""

from os import listdir, makedirs, path
from multiprocessing import Pool, cpu_count
import shutil
import sys

pricer_dir = path.dirname(path.dirname(__file__))
if not pricer_dir in sys.path:
    sys.path.append(pricer_dir)
del pricer_dir

import cv2

from detect_pricer import detect_pricer, detect_pricer_test
from utils.os_utils import ask_dir

def process_file(file, directory, batch_dir, batch_dir_failed):
        if file.endswith("jpg"):       
            img = cv2.imread(directory + "/" + file)
            img = cv2.resize(img,(576, 1024), interpolation = cv2.INTER_CUBIC)
            rects = None
            try:
                rects = detect_pricer_test(img)
            except:
                return
            for rect in rects:
                cv2.rectangle(img, 
                              (int(rect[0]),int(rect[1])), 
                              (int(rect[0] + rect[2]), 
                               int(rect[1] + rect[3])),
                              (255, 0, 0), 3)
                
                x, y = int(img.shape[1] / 2), int(img.shape[0] / 2)
                cv2.rectangle(img, (x - 20, y - 20),
                                   (x + 20, y + 20),
                              (255, 255, 0), 3)
            if(len(rects) == 0):
                cv2.imwrite(batch_dir_failed + "/" + file, img);
            else:                  
                cv2.imwrite(batch_dir + "/" + file, img);


if __name__ == '__main__':
    directory = ask_dir()
    batch_dir = directory + "/batch"
    batch_dir_failed = batch_dir + "/failed"
    
    if(path.exists(batch_dir)):
        shutil.rmtree(batch_dir)
    makedirs(batch_dir)
    makedirs(batch_dir_failed)
    
    def process_file_args(file):
        return process_file(file, directory, batch_dir, batch_dir_failed)

    try:
        pool = Pool(processes=cpu_count())
        pool.map(process_file_args, listdir(directory))
    finally:
        pool.close()
        pool.join()
