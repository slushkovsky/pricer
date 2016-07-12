#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 19:23:15 2016

@author: chernov
"""

from os import environ, system, path, listdir, makedirs
import sys
from argparse import ArgumentParser

import cv2

pricer_dir = path.dirname(path.dirname(__file__))
if not pricer_dir in sys.path:
    sys.path.append(pricer_dir)
del pricer_dir

from split_hist import split_lines_hist, crop_regions, get_begin_end_text_line

DATASET_NAME = "names"
DATA_PATH = environ["BEORGDATAGEN"] + "/CData_full"
DATASET_PATH = DATA_PATH + "/" + DATASET_NAME
OUT_PATH = DATA_PATH + "/names_lines"


def parse_args():
    def file_arg(value): 
        if not path.exists(value):
            if not path.exists(value):
                raise Exception() 
        return value
        
    parser = ArgumentParser()
    parser.add_argument('--path', type=file_arg,
                        help="Image filepath or path to folder with images")
    return parser.parse_args()
    
    
if __name__ == '__main__':
    
    args = parse_args()
    if args.path:
        imgs = None
        folder = None
        if path.isdir(args.path):
            imgs = listdir(args.path)
            folder = args.path
        else:
            imgs = [path.basename(args.path)]
            folder = path.dirname(path.realpath(args.path))
        
        for img_name in imgs:
            img_path = path.join(folder,img_name)
            print(img_path)
            
            img = cv2.imread(img_path)
            
            regions = crop_regions(split_lines_hist(img), img.shape[0]*0.15)
            for region in regions:
                line = img[region[0]:region[1], :]
                regions_x = get_begin_end_text_line(line, test=True)
                cv2.rectangle(img, (0,region[0]), 
                              (img.shape[1],region[1]), (0,0,255), 3)
                for region_x in regions_x:
                    cv2.rectangle(img, (region_x[0], region[0]), 
                                  (region_x[1], region[1]), (255,0,0, 100), 2)
                
            cv2.imshow("img", cv2.resize(img, (600, 200)))
    
            k = cv2.waitKey() & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
         
    else:
        if not path.exists(DATA_PATH):
            print("%s not found, start croping"%(DATA_PATH))
            system("python3 %s/marking_tools/crop_pricers.py"%
                   (path.realpath(path.dirname(path.dirname(__file__)))))
            print("croping done")
        else:
            print("%s found, skip croping. To force restart delete this folder"
                  %(DATA_PATH))
            
        if not path.exists(DATASET_PATH):
            print("%s not found, start croping"%(DATASET_PATH))
            system("python3 %s/marking_tools/crop_pricer_fields.py"%
                   (path.realpath(path.dirname(path.dirname(__file__)))))
            print("croping done")
        
        if(path.exists(OUT_PATH)):
            print("%s already exists, to force crop remove it"%(OUT_PATH))
        else:
            makedirs(OUT_PATH)    
            for img_name in listdir(DATASET_PATH):
                if img_name.endswith("jpg"):
                    img_path = path.join(DATASET_PATH, img_name)
                    img = cv2.imread(img_path)
                    regions = crop_regions(split_lines_hist(img),
                                           img.shape[0]*0.15)
                    for i in range(len(regions)):
                        cur_img_part = img[regions[i][0]:regions[i][1],:]
                        cv2.imwrite(path.join(OUT_PATH, "%i_%s"%(i, img_name)),
                                    cur_img_part)
                    
            
            
            
        
    