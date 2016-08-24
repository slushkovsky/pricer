#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:57:35 2016

@author: chernov
"""

import base64
import sys
from os import path
from argparse import ArgumentParser

import cv2
import numpy as np
from flask import Flask, jsonify, abort, request

main_dir = path.abspath(path.join(path.dirname(__file__), "../"))
if not main_dir in sys.path:
    sys.path.append(main_dir)
del main_dir

from symbol_split_utils.cv_standalone.process_pricer import process_pricer
from symbol_split_utils.cv_standalone.process_pricer import extract_text_from_rects
from beorg_net_utils import Localisator
from symbol_split_utils.cv_standalone.classify_symb import PriceClassifier 
from symbol_split_utils.cv_standalone.classify_symb import SymbolsClassifier

loc_name, loc_rub, loc_kop = None, None, None
name_recog, digit_recog = None, None
args = None

def parse_args():
    def file_arg(value): 
        if not path.exists(value):
            if not path.exists(value):
                raise Exception() 
        return value
        
    nets_path = path.join(path.abspath(path.join(__file__, "../..")),
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
    
    params_group = parser.add_argument_group("Algoritm Parameters")
    params_group.add_argument("--min_var", type=int, default=-1,
                              help="Minimal variance of pricer name field. -1 "
                              "if disabled (default - (-1)).")
    params_group.add_argument("--min_symb_var", type=int, default=1000,
                              help="Minimal variance of pricer (default 1000).")
    
    
    ml_group = parser.add_argument_group("ML Pretrained Files")
    
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
                          help="Path digit classify net .prototxt "
                          "(default - %s)."%(class_dig_proto_def))
    ml_group.add_argument('--digits_net_weights', default=class_dig_weights_def,
                          help="Path digit classify net .caffemodel "
                          "(default - %s)."%(class_dig_weights_def))
    ml_group.add_argument('--digits_net_dict', default=class_dig_dict_def,
                          help="Path digit classify net symbols dictionary"
                          "(default - %s)."%(class_dig_dict_def))  
    
    
    ml_group.add_argument('--nm1', type=file_arg, default=nm1_def,
                          help="Path to pretrained NM1 dtree classifier "
                          "(default - %s)."%(nm1_def))
    ml_group.add_argument('--nm2', type=file_arg, default=nm2_def,
                          help="Path to pretrained NM2 dtree classifier "
                          "(default - %s)."%(nm2_def))

    return parser.parse_args()

app = Flask(__name__)


@app.route('/price', methods=['POST'])
def process_pricer_():
    if not request.json or not "id" in request.json or \
       not "base64String" in request.json:
        abort(400)
    
    nparr = np.fromstring(base64.b64decode(request.json["base64String"]),
                                           np.uint8)
        
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    name_cnt = loc_name.predict(img)
    rub_cnt = loc_rub.predict(img)
    kop_cnt = loc_kop.predict(img)
    
    strings, space_idxs, undef_idxs, \
    rects_rub, rects_kop = process_pricer(img, args.nm1, args.nm2,
                                          name_cnt, rub_cnt, kop_cnt,
                                          min_var=args.min_var,
                                          min_symb_var=args.min_symb_var)
    
    name = extract_text_from_rects(img, strings, space_idxs,
                                   undef_idxs, name_recog)
    rub = digit_recog.covert_rects_to_price(img, rects_rub)
    kop = digit_recog.covert_rects_to_price(img, rects_kop)   
    return jsonify({"name": name, "rub": rub,
                    "kop": kop, "id": request.json["id"]}), 201


if __name__ == '__main__':
    args = parse_args()
    
    loc_name = Localisator(args.name_loc_model, args.name_loc_weights)
    loc_rub = Localisator(args.rub_loc_model, args.rub_loc_weights)
    loc_kop = Localisator(args.kop_loc_model, args.kop_loc_weights)
    
    name_recog = SymbolsClassifier(args.name_net_model,
                                       args.name_net_weights,
                                       args.name_net_dict)
    digit_recog = PriceClassifier(args.digits_net_model,
                                  args.digits_net_weights,
                                  args.digits_net_dict)
    
    app.run(debug=True)
    