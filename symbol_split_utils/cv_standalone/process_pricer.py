#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:13:47 2016

@author: chernov
"""

import glob
import sys
import shutil
from os import path, makedirs
from argparse import ArgumentParser

import cv2
import numpy as np

main_dir = path.abspath(path.join(path.dirname(__file__), "../.."))
if not main_dir in sys.path:
    sys.path.append(main_dir)
del main_dir

from split_hist_cv import split_lines_hist, crop_regions
from split_symbols_cv import detect_text_cv
from split_ruble_symbols import process_image

def resize_img_h(img, new_h):
    scale = new_h / img.shape[0]
    new_size = np.array(img.shape) * scale 
    new_size = tuple(np.flipud(new_size[0:2]).astype(np.int))
    new_img = cv2.resize(img, new_size)
    return new_img, scale
    
    
def parse_mark(line, datapath, prefix=""):
    s = line.split()
    img = cv2.imread(path.join(datapath, prefix + s[0]))
    
    cnt = np.zeros(8, np.float)
    for i in range(1, 9, 2):
        cnt[i - 1] = float(s[i])
        cnt[i] = float(s[i + 1])
    cnt = cnt.reshape((4,2)).astype(np.int)
    return img, cnt
    
    
def process_pricer(datapath, line, prefix="", new_h=700):
    img, cnt = parse_mark(line, datapath, prefix=prefix)
    if img is None:
        return None, cnt
    img, scale = resize_img_h(img, new_h=new_h)
    cnt = (cnt * scale).astype(np.int)
    return img, cnt, scale
    
    
def extract_symbol_rects(crop, nm1, nm2, offset=(0,0), min_variance=1000):
    rects, rects_bad = [], []
    regions = [[0,  crop.shape[0]]]#crop_regions(split_lines_hist(crop), crop.shape[0]*0.15)   
    for i in range(len(regions)):
        cur_img_part = crop[regions[i][0]:regions[i][1],:]
        cur_rects, cur_rects_bad = detect_text_cv(cur_img_part,
                                                  nm1, 
                                                  nm2,
                                                  min_variance=min_variance)
        
        for rect in cur_rects:
            rect[1] += regions[i][0] + offset[1]
            rect[0] += offset[0]
            rects.append(rect)
            
        for rect in cur_rects_bad:
            rect[1] += regions[i][0] + offset[1]
            rect[0] += offset[0]
            rects_bad.append(rect)
    return rects, rects_bad
    
    
def extract_rubli_rects(img, offset=(0,0)):
    rects = process_image(img)
    for i in range(len(rects)):
        rects[i][0] += offset[0]
        rects[i][1] += offset[1]
    return rects
    
    
def merge_mark_files(files):
    markings = dict()
    for file in files:
        for line in open(file, "r").readlines():
            markings[line.split()[0]] = line
    return markings

    
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()    

    
def parse_args():
    def file_arg(value): 
        if not path.exists(value):
            if not path.exists(value):
                raise Exception() 
        return value
        
    parser = ArgumentParser()
    parser.add_argument('datapath', type=file_arg,
                        help="Path to folder with image data")
    parser.add_argument('names_wildcard', type=str,
                        help="Wildcard expression with names marks files")
    parser.add_argument('rubli_wildcard', type=str,
                        help="Wildcard expression with rubles marks files")
    parser.add_argument('--nm1', type=file_arg,
                        default=path.abspath(path.join(path.dirname(__file__), 
                                          "pretrained_classifiers/"
                                          "trained_classifierNM1.xml")),
                        help="Path to pretrained NM1 dtree classifier")
    parser.add_argument('--nm2', type=file_arg,
                        default=path.abspath(path.join(path.dirname(__file__), 
                                          "pretrained_classifiers/"
                                          "trained_classifierNM2.xml")),
                        help="Path to pretrained NM2 dtree classifier")
    parser.add_argument("--img", help="execute for one image")
    parser.add_argument("--outdir", default=path.join(path.dirname(__file__),
                                                      "split_data"),
                        help="output directory")
    parser.add_argument("--resize_out", action="store_true",
                        help="Resize output marked pricers")
    parser.add_argument("--resize_h", type=int, default=700,
                        help="Internal processing pricer height size "
                        "(default 700)")
    parser.add_argument("--min_var", type=int, default=-1,
                        help="Minimal variance of pricer name field")
    parser.add_argument("--min_symb_var", type=int, default=1000,
                        help="Minimal variance of pricer (default 1000)")
                        
    return parser.parse_args()
    
    
if __name__ == "__main__":
    args = parse_args()
    
    names_marks = merge_mark_files(glob.glob(args.names_wildcard))
    rubles_marks = merge_mark_files(glob.glob(args.rubli_wildcard))
    
    keys = []
    keys += list(names_marks.keys())
    keys +=list(rubles_marks.keys())
    keys = list(set(keys))
    
    batch = []
    if args.img:
        batch.append(args.img)
    else:
        batch += keys
        if path.exists(args.outdir):
            shutil.rmtree(args.outdir)
        makedirs(args.outdir)
    
    for key in batch:
        rects, rects_bad, rects_rub = [], [], []
        scale = None
        if not path.exists(path.join(args.datapath, key)):
            print("image %s not found"%(path.join(args.datapath, key)))
            continue
        
        scale = 1
        img = None
        if key in names_marks:
            try:
                img, cnt, scale = process_pricer(args.datapath,
                                                 names_marks[key],
                                                 new_h=args.resize_h)
                
                if img is None:
                    print("cant open image %s"%(path.join(args.datapath, key)))
                    continue
                x,y,w,h = cv2.boundingRect(cnt)
                
                crop = img[y:y+h, x:x+w]

                if args.min_var > 0:
                    var = variance_of_laplacian(crop)
                    if args.img:
                        print("variance - %s"%(var))
                    if var < args.min_var:
                        print("%s too blured (%s < %s)"%(key, var,
                              args.min_var))
                        continue
            
                rects, rects_bad = extract_symbol_rects(crop, args.nm1,
                                                        args.nm2,
                                                        offset=(x,y), 
                                                        min_variance=\
                                                        args.min_symb_var)
            except ValueError:
                print("cant process image %s"%(path.join(args.datapath, key)))
                continue
        elif args.min_var > 0:
            continue
            
        if key in rubles_marks:
            try:
                img, cnt, scale = process_pricer(args.datapath,
                                                 rubles_marks[key], 
                                                 new_h=args.resize_h)
                if img is None:
                    print("cant open image %s"%(path.join(args.datapath, key)))
                    continue
                x,y,w,h = cv2.boundingRect(cnt)
                crop = img[y:y+h, x:x+w]
                rects_rub = extract_rubli_rects(crop, offset=(x,y))
            except ValueError:
                print("cant process image %s"%(path.join(args.datapath, key)))
                continue
        
        vis = None
        if args.resize_out:
            vis = img.copy()
        else:
            vis = cv2.imread(path.join(args.datapath, key))
            rects_rub = [(np.array(r)/scale).astype(np.int) for r in rects_rub]
            rects = [(np.array(r)/scale).astype(np.int) for r in rects]
        
        for rect in rects_rub:
            cv2.rectangle(vis, (rect[0], rect[1]),
                          (rect[0] + rect[2], rect[1] + rect[3]),
                          (0, 255, 0), 2)
       
        for rect in rects:
            cv2.rectangle(vis, (rect[0], rect[1]),
                          (rect[0] + rect[2], rect[1] + rect[3]),
                          (0, 0, 255), 2)
        if args.img:
            cv2.imshow("vis", vis)
            cv2.waitKey()
            cv2.destroyAllWindows()
            break
        else:
            cv2.imwrite(path.join(args.outdir, key), vis)
            if not args.img:
                out = open(path.join(args.outdir, 
                                     path.splitext(path.basename(key))[0] + 
                                     ".box"), "w")
            for rect in rects:
                out.write("* %s %s %s %s 0 n\n"%(int(rect[0]/scale), 
                                                 int((rect[1] + rect[3])/scale), 
                                                 int((rect[0] + rect[2])/scale), 
                                                 int((rect[1])/scale)))
            for rect in rects_rub:
                out.write("* %s %s %s %s 0 r\n"%(int(rect[0]/scale), 
                                                 int((rect[1] + rect[3])/scale), 
                                                 int((rect[0] + rect[2])/scale), 
                                                 int((rect[1])/scale)))
            
        
        
        
        
