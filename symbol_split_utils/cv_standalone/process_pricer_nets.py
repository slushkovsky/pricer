#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 20:25:02 2016

@author: chernov
"""

import sys
import json
import shutil
from os import path, makedirs, listdir
from argparse import ArgumentParser

import cv2
import numpy as np

main_dir = path.abspath(path.join(path.dirname(__file__), "../.."))
if not main_dir in sys.path:
    sys.path.append(main_dir)
del main_dir
cur_dir = path.abspath(path.dirname(__file__))
if not cur_dir in sys.path:
    sys.path.append(cur_dir)
del cur_dir

from server.beorg_net_utils import Localisator
from classify_symb import PriceClassifier, SymbolsClassifier
from process_pricer import process_pricer, resize_h, extract_text_from_rects
from process_pricer import get_rect_from_cnt

def parse_args():
    def file_arg(value): 
        if not path.exists(value):
            if not path.exists(value):
                raise Exception() 
        return value
        
    nets_path = path.join(path.abspath(path.join(__file__, "../../..")),
                          "ml_data")
    
    loc_name_path = path.join(nets_path, "loc_names")
    name_loc_proto_def = path.join(loc_name_path, "names_net.prototxt")
    name_loc_weights_def = path.join(loc_name_path, "names_net.caffemodel")
    
    loc_rub_path = path.join(nets_path, "loc_rubles")
    rub_loc_proto_def = path.join(loc_rub_path, "rubles_net.prototxt")
    rub_loc_weights_def = path.join(loc_rub_path, "rubles_net.caffemodel")
    
    loc_kop_path = path.join(nets_path, "loc_kopecks")
    kop_loc_proto_def = path.join(loc_kop_path, "kopecks_net.prototxt")
    kop_loc_weights_def = path.join(loc_kop_path, "kopecks_net.caffemodel")
    
    class_dig_path = path.join(nets_path, "class_digits")
    class_dig_proto_def = path.join(class_dig_path, "digits_net.prototxt")
    class_dig_weights_def = path.join(class_dig_path, "digits_net.caffemodel")
    class_dig_dict_def = path.join(class_dig_path, "digits_net_dict.json")
    
    class_symb_path = path.join(nets_path, "class_symb")
    class_symb_proto_def = path.join(class_symb_path, "symbols_net.prototxt")
    class_symb_weights_def = path.join(class_symb_path, "symbols_net.caffemodel")
    class_symb_dict_def = path.join(class_symb_path, "symbols_net_dict.json")
    
    nm_path = path.join(nets_path, "NM")
    nm1_def = path.join(nm_path,"trained_classifierNM1.xml")
    nm2_def = path.join(nm_path,"trained_classifierNM1.xml")    
        
    parser = ArgumentParser()
    
    data_group = parser.add_argument_group("Data Settings")
    data_group.add_argument('datapath', type=file_arg,
                            help="Path to folder with image data.")
    data_group.add_argument("--img", help="Execute for single image in dataset.")

    params_group = parser.add_argument_group("Algoritm Parameters")
    params_group.add_argument("--resize_h", type=int, default=700,
                               help="Internal processing pricer height size "
                               "(default - 700).")
    params_group.add_argument("--min_var", type=int, default=-1,
                              help="Minimal variance of pricer name field. -1 "
                              "if disabled (default - (-1)).")
    params_group.add_argument("--min_symb_var", type=int, default=200,
                              help="Minimal variance of pricer (default 200).")
    
    ml_group = parser.add_argument_group("ML Pretrained Files")
    ml_group.add_argument('--nm1', type=file_arg, default=nm1_def,
                          help="Path to pretrained NM1 dtree classifier "
                          "(default - %s)."%(nm1_def))
    ml_group.add_argument('--nm2', type=file_arg, default=nm2_def,
                          help="Path to pretrained NM2 dtree classifier "
                          "(default - %s)."%(nm2_def))
    
    ml_group.add_argument('--name_net_model', default=class_symb_proto_def,
                          help="Path symbol classify net .prototxt "
                          "(default - %s)."%(class_symb_proto_def))
    ml_group.add_argument('--name_net_weights', default=class_symb_weights_def,
                          help="Path symbol classify net .caffemodel "
                          "(default - %s)."%(class_symb_weights_def))
    ml_group.add_argument('--name_net_dict', default=class_symb_dict_def,
                          help="Path symbol classify net symbols dictionary"
                          "(default - %s)."%(class_symb_dict_def))
    
    ml_group.add_argument('--digits_net_model', default=class_dig_proto_def,
                          help="Path symbol classify net .prototxt "
                          "(default - %s)."%(class_dig_proto_def))
    ml_group.add_argument('--digits_net_weights', default=class_dig_weights_def,
                          help="Path symbol classify net .caffemodel "
                          "(default - %s)."%(class_dig_weights_def))
    ml_group.add_argument('--digits_net_dict', default=class_dig_dict_def,
                          help="Path symbol classify net symbols dictionary"
                          "(default - %s)."%(class_dig_dict_def))
    
    ml_group.add_argument('--name_loc_model', default=name_loc_proto_def,
                          help="Path to name localistion net .prototxt "
                          "(default - %s)."%(name_loc_proto_def))
    ml_group.add_argument('--name_loc_weights', default=name_loc_weights_def,
                          help="Path to name localistion net .caffemodel "
                          "(default - %s)."%(name_loc_weights_def))
    
    ml_group.add_argument('--rub_loc_model', default=rub_loc_proto_def,
                          help="Path to ruble localistion net .prototxt "
                          "(default - %s)."%(rub_loc_proto_def))
    ml_group.add_argument('--rub_loc_weights', default=rub_loc_weights_def,
                          help="Path to ruble localistion net .caffemodel "
                          "(default - %s)."%(rub_loc_weights_def))

    ml_group.add_argument('--kop_loc_model', default=kop_loc_proto_def,
                          help="Path to kopeck localistion net .prototxt "
                          "(default - %s)."%(kop_loc_proto_def))
    ml_group.add_argument('--kop_loc_weights', default=kop_loc_weights_def,
                          help="Path to kopeck localistion net .caffemodel "
                          "(default - %s)."%(kop_loc_weights_def))
    
    output_group = parser.add_argument_group("Output Settings")
    out_def = path.join(path.dirname(__file__),"split_data_nets")
    output_group.add_argument("--outdir", default=out_def,
                              help="Output directory (default - %s)."%(out_def))
    
    output_group.add_argument("--out_vis", action="store_true",
                               help="Save images with marks.")
    
    output_group.add_argument("--out_vis_nobounds", action="store_true",
                               help="Dont draw fields bounds on out image.")
    output_group.add_argument("--out_vis_nosymbols", action="store_true",
                               help="Dont draw symbols boxes on out image.")
    output_group.add_argument("--out_vis_nodigits", action="store_true",
                               help="Dont draw digits boxes on out image.")
    
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

    
if __name__ == "__main__":
    args = parse_args()
    
    loc_name = Localisator(args.name_loc_model, args.name_loc_weights)
    loc_rub = Localisator(args.rub_loc_model, args.rub_loc_weights)
    loc_kop = Localisator(args.kop_loc_model, args.kop_loc_weights)
    
    name_recog, digit_recog  = None, None
    if args.out_json:
        name_recog = SymbolsClassifier(args.name_net_model,
                                       args.name_net_weights,
                                       args.name_net_dict)
        digit_recog = PriceClassifier(args.digits_net_model,
                                      args.digits_net_weights,
                                      args.digits_net_dict)
    
    batch = []
    if args.img:
        batch.append(args.img)
    else:
        batch += listdir(args.datapath)
        
    if args.out_box or args.out_json or args.out_vis:
        if path.exists(args.outdir):
            shutil.rmtree(args.outdir)
        makedirs(args.outdir)
    
    for key in batch:
        
        if not path.exists(path.join(args.datapath, key)):
            print("image %s not found"%(path.join(args.datapath, key)))
            continue
        
        img = cv2.imread(path.join(args.datapath, key), 0)
        if img is None:
            print("cant open %s"%(path.join(args.datapath, key)))
            continue
        
        name_cnt = loc_name.predict(img)
        rub_cnt = loc_rub.predict(img)
        kop_cnt = loc_kop.predict(img)
        
        try:
            strings, space_idxs, undef_idxs, \
            rects_rub, rects_kop = process_pricer(img, args.nm1, args.nm2,
                                                  name_cnt, rub_cnt, kop_cnt,
                                                  min_var=args.min_var,
                                                  min_symb_var=
                                                  args.min_symb_var)
        except Exception as e: 
            print("%s Exception: %s"%(key, e))
            continue 
        
        
        img = cv2.imread(path.join(args.datapath, key))
        vis = img
        if args.resize_out:
            vis, scale = resize_h(img, args.resize_h)
            img = vis
            rects_kop = [(np.array(r)*scale).astype(np.int) for r in rects_kop]
            rects_rub = [(np.array(r)*scale).astype(np.int) for r in rects_rub]
            for i in range(len(strings)):
                strings[i] = [(np.array(r)*scale).astype(np.int) for r \
                              in strings[i]]

        # объединение отфильтрованных символов
        if args.out_box:
            out = open(path.join(args.outdir, 
                                 path.splitext(path.basename(key))[0] + 
                                 ".box"), "w")
            batch = {"n": strings,
                     "r": rects_rub, "k": rects_kop}
            for ele in batch:
                for rect in batch[ele]:
                    out.write("* %s %s %s %s 0 %s\n"%(int(rect[0]), 
                                                     int(rect[1] + rect[3]), 
                                                     int(rect[0] + rect[2]), 
                                                     int(rect[1]), ele))
            out.close()
            
        if args.out_json:
            dict_ = {"name": "", "rub": -1, "kop": -1}

            dict_["name"] = extract_text_from_rects(img, strings, space_idxs,
                                                    undef_idxs, name_recog)
            if len(rects_rub) > 0:
                dict_["rub"] = digit_recog.covert_rects_to_price(img, 
                                                                 rects_rub)
            if len(rects_kop) > 0:
                dict_["kop"] = digit_recog.covert_rects_to_price(img, 
                                                                 rects_kop)
                
            json.dump(dict_, open(path.join(args.outdir,
                                            path.splitext(path.basename(key))[0] +
                                  ".json"), "w"),
                      ensure_ascii=False)
                
        if args.out_vis:
            if not args.out_vis_nobounds:   
                for rect in [get_rect_from_cnt(name_cnt, img),
                             get_rect_from_cnt(rub_cnt, img), 
                             get_rect_from_cnt(kop_cnt, img)]:
                    if rect:
                        cv2.rectangle(vis, (rect[0], rect[1]), 
                                      (rect[0] + rect[2], 
                                       rect[1] + rect[3]), 
                                      (255, 125, 0), 1)

            if not args.out_vis_nosymbols:
                spaces = np.array(strings)[space_idxs]
                undefs = np.array(strings)[undef_idxs]
                symbols = np.delete(strings,
                                    np.append(space_idxs, undef_idxs), axis=0)
                
                for rects, color in [[symbols, (255,0,0)], [spaces,
                                     (0,255,255)],
                                     [undefs, (255, 0,255)]]:
                    for rect in rects:
                        cv2.rectangle(vis, (rect[0], rect[1]),
                                      (rect[0] + rect[2], rect[1] + rect[3]),
                                      color, 1)
                        
            if not args.out_vis_nodigits:
                for rects in [rects_kop, rects_rub]:
                    for rect in rects:
                        cv2.rectangle(vis, (rect[0], rect[1]),
                                      (rect[0] + rect[2], rect[1] + rect[3]),
                                      (0, 255, 0), 1)
                        
            cv2.imwrite(path.join(args.outdir, key), vis)
            
        if args.img:
            cv2.imshow("vis", vis)
            cv2.waitKey()
            cv2.destroyAllWindows()