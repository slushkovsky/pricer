#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 00:25:37 2016

@author: chernov
"""

import os
from argparse import ArgumentParser
import shutil

import cv2
import numpy as np

from crop_pricer_quadrants import crop_quadrant
from increase_data import increase_data

def parse_args():
    def file_arg(value): 
        if not os.path.exists(value):
            if not os.path.exists(value):
                raise Exception() 
        return value
        
    parser = ArgumentParser()
    parser.add_argument('markings', type=file_arg,
                        help="Path to file with markings. "
                        "Image directory should ./images relative to this "
                        "file")
    parser.add_argument('out', type=str,
                        help="Output path")
    parser.add_argument('--increase', type=int, default=10,
                        help="Increase rate (default - 10)")
    parser.add_argument('--increase_art', type=int, default=2,
                        help="Artifial increase (smooth, change colors)"
                        " data rate (default - 2)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    inc_rate = args.increase
    
    markings_file = args.markings
    image_path = os.path.join(os.path.dirname(markings_file), "images")
    
    out_path = args.out
    out_path_images = os.path.join(out_path, "images")
    out_path_marks = os.path.join(out_path, "corners.txt")
    if(os.path.exists(out_path)):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    os.makedirs(out_path_images)
    
    with open(markings_file, "r") as f:
        with open(out_path_marks, "w") as f_out:
            for i, line in enumerate(f.readlines()):
                vals = line.split()
                
                filepath = os.path.join(image_path, vals[0])
                img = cv2.imread(filepath)
                
                rect = np.array(([int(vals[1]), int(vals[2])],
                                 [int(vals[3]), int(vals[4])],
                                 [int(vals[5]), int(vals[6])],
                                 [int(vals[7]), int(vals[8])]))
                
                for j in range(inc_rate):
                    warped, corner = crop_quadrant(img, rect)
                    
                    if type(warped) == type(None):
                        print("%s rect finding failed"%(filepath))
                        continue
                    
                    art_inc_imgs = increase_data(warped, args.increase_art)
                    art_inc_imgs.append(warped)
                    
                    for k, img_cur in enumerate(art_inc_imgs):
                        name = "%s_%s_%s"%(j, k, vals[0])
                        cv2.imwrite(os.path.join(out_path_images, name),
                                    img_cur)
                        f_out.write("%s %s %s\n"%(name, int(corner[0]),
                                                        int(corner[1])))
                    
    
    