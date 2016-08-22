#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:13:47 2016

@author: chernov
"""

import glob
import sys
import json
import shutil
from os import path, makedirs
from argparse import ArgumentParser

import cv2
import numpy as np

main_dir = path.abspath(path.join(path.dirname(__file__), "../.."))
if not main_dir in sys.path:
    sys.path.append(main_dir)
del main_dir

from classify_symb import PriceClassifier, NameClassifier
from split_symbols_cv import detect_text_cv, normalize_crop
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


def extract_price_rects(img, offset=(0,0)):
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
	return cv2.Laplacian(image, cv2.CV_64F).var()    


def parse_args():
    def file_arg(value): 
        if not path.exists(value):
            if not path.exists(value):
                raise Exception() 
        return value
        
    parser = ArgumentParser()
    
    data_group = parser.add_argument_group("Data Settings")
    data_group.add_argument('datapath', type=file_arg,
                            help="Path to folder with image data.")
    data_group.add_argument('names_wildcard', type=str,
                            help="Wildcard expression with names marks files.")
    data_group.add_argument('rubli_wildcard', type=str,
                            help="Wildcard expression with rubles marks files.")
    data_group.add_argument('kopeck_wildcard', type=str,
                            help="Wildcard expression with kopecks marks files.")
    data_group.add_argument("--img", help="Execute for single image in dataset.")
    
    params_group = parser.add_argument_group("Algoritm Parameters")
    params_group.add_argument("--resize_h", type=int, default=700,
                               help="Internal processing pricer height size "
                               "(default - 700).")
    params_group.add_argument("--min_var", type=int, default=-1,
                              help="Minimal variance of pricer name field. -1 "
                              "if disabled (default - (-1)).")
    params_group.add_argument("--min_symb_var", type=int, default=1000,
                              help="Minimal variance of pricer (default 1000).")
    
    ml_group = parser.add_argument_group("ML Pretrained Files")
    nm1_default = path.abspath(path.join(path.dirname(__file__),
                                         "pretrained_classifiers/"
                                         "trained_classifierNM1.xml"))
    ml_group.add_argument('--nm1', type=file_arg,
                          default=nm1_default,
                          help="Path to pretrained NM1 dtree classifier "
                          "(default - %s)."%(nm1_default))
    nm2_default = path.abspath(path.join(path.dirname(__file__),
                                         "pretrained_classifiers/"
                                         "trained_classifierNM2.xml"))
    ml_group.add_argument('--nm2', type=file_arg,
                          default=nm2_default,
                          help="Path to pretrained NM2 dtree classifier "
                          "(default - %s)."%(nm2_default))
    name_net_proto_def = path.abspath(path.join(path.dirname(__file__),
                                                "pretrained_classifiers/"
                                                "symbols_net.prototxt"))
    ml_group.add_argument('--name_net_model', type=file_arg,
                          default=name_net_proto_def,
                          help="Path symbol classify net .prototxt "
                          "(default - %s)."%(name_net_proto_def))
    name_net_weights_def = path.abspath(path.join(path.dirname(__file__),
                                                  "pretrained_classifiers/"
                                                  "symbols_iter_10000.caffemodel"))
    ml_group.add_argument('--name_net_weights', type=file_arg,
                          default=name_net_weights_def,
                          help="Path symbol classify net .caffemodel "
                          "(default - %s)."%(name_net_proto_def))
    name_net_dict_def = path.abspath(path.join(path.dirname(__file__),
                                               "pretrained_classifiers/"
                                               "symbols_lmdb_dict.json"))
    ml_group.add_argument('--name_net_dict', type=file_arg,
                          default=name_net_dict_def,
                          help="Path symbol classify net symbols dictionary"
                          "(default - %s)."%(name_net_dict_def))
    dig_net_proto_def = path.abspath(path.join(path.dirname(__file__),
                                               "pretrained_classifiers/"
                                               "price_1_net.prototxt"))
    ml_group.add_argument('--digits_net_model', type=file_arg,
                          default=dig_net_proto_def,
                          help="Path symbol classify net .prototxt "
                          "(default - %s)."%(name_net_proto_def))
    dig_net_weights_def = path.abspath(path.join(path.dirname(__file__),
                                                 "pretrained_classifiers/"
                                                 "price_1_iter_1000.caffemodel"))
    ml_group.add_argument('--digits_net_weights', type=file_arg,
                          default=dig_net_weights_def,
                          help="Path symbol classify net .caffemodel "
                          "(default - %s)."%(name_net_proto_def))
    dig_net_dict_def = path.abspath(path.join(path.dirname(__file__),
                                              "pretrained_classifiers/"
                                              "price_1_lmdb_dict.json"))
    ml_group.add_argument('--digits_net_dict', type=file_arg,
                          default=dig_net_dict_def,
                          help="Path symbol classify net symbols dictionary"
                          "(default - %s)."%(name_net_dict_def))
    
    output_group = parser.add_argument_group("Output Settings")
    out_def = path.join(path.dirname(__file__),"split_data")
    output_group.add_argument("--outdir", default=out_def,
                              help="Output directory (default - %s)."%(out_def))
    output_group.add_argument("--out_vis", action="store_true",
                               help="Save images with marks.")
    output_group.add_argument("--resize_out", action="store_true",
                               help="Resize output image. If enabled, resised "
                               "height will be equal --resize_h value.")
    output_group.add_argument("--out_box", action="store_true",
                              help="Save symbol markings to .box file")
    output_group.add_argument("--out_json", action="store_true",
                              help="Save output to json file. This flag will "
                              "activate symbol classification step. Proper "
                              "name net parameters (--name_net_dict, "
                              "--name_net_weights, --name_net_model) is needed")
    return parser.parse_args()
    
    
def open_pricer(datapath, markings, new_h):
    try:
        img, cnt, scale = process_pricer(datapath,
                                         markings,
                                         new_h=new_h)
        return img, cnt, scale
    except ValueError:
        print("cant process image %s"%(path.join(datapath, key)))
        return None, None, None
       
 
def process_name(datapath, key,  markings, new_h):
    img, cnt, scale = open_pricer(datapath, markings, new_h)
        
    if img is None:
        print("cant open image %s"%(path.join(args.datapath, key)))
        return [], None, img
        
    name_rect = cv2.boundingRect(cnt)
    crop = img[name_rect[1]:name_rect[1]+name_rect[3],
               name_rect[0]:name_rect[0]+name_rect[2]]
    
    if args.min_var > 0:
        var = variance_of_laplacian(crop)
        if args.img:
            print("variance - %s"%(var))
        if var < args.min_var:
            raise Exception("too blured (%s < %s)"%(var, args.min_var))
            
    img[name_rect[1]:name_rect[1]+name_rect[3],
        name_rect[0]:name_rect[0]+name_rect[2]] = normalize_crop(crop)
    
    try:
        rects, rects_bad = extract_symbol_rects(crop, args.nm1,
                                                args.nm2,
                                                offset=(name_rect[0],
                                                        name_rect[1]), 
                                                min_variance=\
                                                args.min_symb_var)
        return rects, name_rect, img
    except ValueError:
        print("cant extract name symbols %s"%(path.join(datapath, key)))
        return [], name_rect, img
    
        
def process_price(datapath, key,  markings, new_h):
    img, cnt, scale = open_pricer(datapath, markings, new_h)
    
    if img is None:
        print("cant open image %s"%(path.join(args.datapath, key)))
        return [], None
    
    price_rect = cv2.boundingRect(cnt)
    crop = img[price_rect[1]:price_rect[1]+price_rect[3],
               price_rect[0]:price_rect[0]+price_rect[2]]

    try:
        rects_price = extract_price_rects(crop, offset=(price_rect[0],
                                                        price_rect[1]))
        return rects_price, price_rect
    except ValueError:
        print("cant process image %s"%(path.join(args.datapath, key)))
        return [], price_rect
    
    
if __name__ == "__main__":
    args = parse_args()
    
    names_marks = merge_mark_files(glob.glob(args.names_wildcard))
    rubles_marks = merge_mark_files(glob.glob(args.rubli_wildcard))
    kopecks_marks = merge_mark_files(glob.glob(args.kopeck_wildcard))
    
    keys = []
    keys += list(names_marks.keys())
    keys +=list(rubles_marks.keys())
    keys +=list(kopecks_marks.keys())
    keys = list(set(keys))
    
    name_recog = None
    digit_recog = None
    if args.out_json:
        name_recog = NameClassifier(args.name_net_model,
                                    args.name_net_weights,
                                    args.name_net_dict)
        digit_recog = PriceClassifier(args.digits_net_model,
                                      args.digits_net_weights,
                                      args.digits_net_dict)
    
    batch = []
    if args.img:
        batch.append(args.img)
    else:
        batch += keys
        
    if args.out_box or args.out_json or args.out_vis:
        if path.exists(args.outdir):
            shutil.rmtree(args.outdir)
        makedirs(args.outdir)
    
    for key in batch:
        rects, rects_rub, rects_kop = [], [], []
        name_rect, rub_rect, kop_rect = None, None, None
        scale = None
        if not path.exists(path.join(args.datapath, key)):
            print("image %s not found"%(path.join(args.datapath, key)))
            continue
        
        scale = 1
        img = None

        if key in names_marks:
            try:
                rects, name_rect, img = process_name(args.datapath, key,
                                                     names_marks[key],
                                                     new_h=args.resize_h)
            except Exception as e:
                print("%s Exception: %s"%(key, e))
                continue
        elif args.min_var > 0:
            continue
            
        if key in rubles_marks:
            rects_rub, rub_rect = process_price(args.datapath, key,
                                                rubles_marks[key],
                                                new_h=args.resize_h)
            
        if key in kopecks_marks:
            rects_kop, kop_rect = process_price(args.datapath, key,
                                                kopecks_marks[key],
                                                new_h=args.resize_h)

        vis = None
        if args.resize_out:
            vis = img.copy()
        else:
            vis = cv2.imread(path.join(args.datapath, key))
            rects_kop = [(np.array(r)/scale).astype(np.int) for r in rects_kop]
            rects_rub = [(np.array(r)/scale).astype(np.int) for r in rects_rub]
            rects = [(np.array(r)/scale).astype(np.int) for r in rects]
        
        if args.out_box:
            out = open(path.join(args.outdir, 
                                 path.splitext(path.basename(key))[0] + 
                                 ".box"), "w")
            batch = {"n": rects, "r": rects_rub, "k": rects_kop}
            for ele in batch:
                for rect in batch[ele]:
                    out.write("* %s %s %s %s 0 %s\n"%(int(rect[0]), 
                                                     int(rect[1] + rect[3]), 
                                                     int(rect[0] + rect[2]), 
                                                     int(rect[1]), ele))
            out.close()
            
        if args.out_json:
            dict_ = {"name": "", "rub": -1, "kop": -1}
            dict_["name"] = name_recog.covert_rects_to_text(img, rects)
            try:
                dict_["rub"] = digit_recog.covert_rects_to_price(img,
                                                                 rects_rub)
                dict_["kop"] = digit_recog.covert_rects_to_price(img,
                                                                rects_kop)
            except:
                pass
            json.dump(dict_, open(path.join(args.outdir,
                                            path.splitext(path.basename(key))[0] +
                                  ".json"), "w"),
                      ensure_ascii=False)
                
        if args.out_vis:
            for rect in [name_rect, rub_rect, kop_rect]:
                if rect:
                    cv2.rectangle(vis, (rect[0], rect[1]), 
                                  (rect[0] + rect[2], 
                                   rect[1] + rect[3]), 
                                  (255, 0, 0), 2)
                              
            for rects in [rects_kop, rects_rub, rects]:
                for rect in rects:
                    cv2.rectangle(vis, (rect[0], rect[1]),
                                  (rect[0] + rect[2], rect[1] + rect[3]),
                                  (0, 255, 0), 2)
                    
            cv2.imwrite(path.join(args.outdir, key), vis)
            
        if args.img:
            cv2.imshow("vis", vis)
            cv2.waitKey()
            cv2.destroyAllWindows()
            