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
from localisation_utils import Localisator
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
        
    parser = ArgumentParser()
    
    params_group = parser.add_argument_group("Algoritm Parameters")
    params_group.add_argument("--min_var", type=int, default=-1,
                              help="Minimal variance of pricer name field. -1 "
                              "if disabled (default - (-1)).")
    params_group.add_argument("--min_symb_var", type=int, default=1000,
                              help="Minimal variance of pricer (default 1000).")
    
    
    ml_group = parser.add_argument_group("ML Pretrained Files")
    
    name_loc_proto_def = path.abspath(path.join(path.dirname(__file__),
                                                "rubli_net/"
                                                "rubli_net.prototxt"))
    
    ml_group.add_argument('--name_loc_model',
                          default=name_loc_proto_def,
                          help="Path to name localistion net .prototxt "
                          "(default - %s)."%(name_loc_proto_def))
    name_loc_weights_def = path.abspath(path.join(path.dirname(__file__),
                                                  "rubli_net/"
                                                  "pricer_rubli2_iter_100000.caffemodel"))
    ml_group.add_argument('--name_loc_weights',
                          default=name_loc_weights_def,
                          help="Path to name localistion net .caffemodel "
                          "(default - %s)."%(name_loc_weights_def))
    
    ml_group.add_argument('--rub_loc_model',
                          default=name_loc_proto_def,
                          help="Path to ruble localistion net .prototxt "
                          "(default - %s)."%(name_loc_proto_def))
    ml_group.add_argument('--rub_loc_weights',
                          default=name_loc_weights_def,
                          help="Path to ruble localistion net .caffemodel "
                          "(default - %s)."%(name_loc_weights_def))

    ml_group.add_argument('--kop_loc_model',
                          default=name_loc_proto_def,
                          help="Path to kopeck localistion net .prototxt "
                          "(default - %s)."%(name_loc_proto_def))
    ml_group.add_argument('--kop_loc_weights',
                          default=name_loc_weights_def,
                          help="Path to kopeck localistion net .caffemodel "
                          "(default - %s)."%(name_loc_weights_def))
    
    
    classif_rel_path = "../symbol_split_utils/cv_standalone/pretrained_classifiers/"
    name_net_proto_def = path.abspath(path.join(path.dirname(__file__),
                                                classif_rel_path,
                                                "symbols_net.prototxt"))
    ml_group.add_argument('--name_net_model',
                          default=name_net_proto_def,
                          help="Path symbol classify net .prototxt "
                          "(default - %s)."%(name_net_proto_def))
    name_net_weights_def = path.abspath(path.join(path.dirname(__file__),
                                                  classif_rel_path,
                                                  "symbols_iter_10000.caffemodel"))
    ml_group.add_argument('--name_net_weights',
                          default=name_net_weights_def,
                          help="Path symbol classify net .caffemodel "
                          "(default - %s)."%(name_net_proto_def))
    name_net_dict_def = path.abspath(path.join(path.dirname(__file__),
                                               classif_rel_path,
                                               "symbols_lmdb_dict.json"))
    ml_group.add_argument('--name_net_dict',
                          default=name_net_dict_def,
                          help="Path symbol classify net symbols dictionary"
                          "(default - %s)."%(name_net_dict_def))
    dig_net_proto_def = path.abspath(path.join(path.dirname(__file__),
                                               classif_rel_path,
                                               "price_1_net.prototxt"))
    ml_group.add_argument('--digits_net_model',
                          default=dig_net_proto_def,
                          help="Path digit classify net .prototxt "
                          "(default - %s)."%(name_net_proto_def))
    dig_net_weights_def = path.abspath(path.join(path.dirname(__file__),
                                                 classif_rel_path,
                                                 "price_1_iter_1000.caffemodel"))
    ml_group.add_argument('--digits_net_weights',
                          default=dig_net_weights_def,
                          help="Path digit classify net .caffemodel "
                          "(default - %s)."%(name_net_proto_def))
    dig_net_dict_def = path.abspath(path.join(path.dirname(__file__),
                                              classif_rel_path,
                                              "price_1_lmdb_dict.json"))
    ml_group.add_argument('--digits_net_dict',
                          default=dig_net_dict_def,
                          help="Path digit classify net symbols dictionary"
                          "(default - %s)."%(name_net_dict_def))  
    
    
    nm1_default = path.abspath(path.join(classif_rel_path,
                                         "trained_classifierNM1.xml"))
    ml_group.add_argument('--nm1', type=file_arg,
                          default=nm1_default,
                          help="Path to pretrained NM1 dtree classifier "
                          "(default - %s)."%(nm1_default))
    nm2_default = path.abspath(path.join(classif_rel_path,
                                         "trained_classifierNM2.xml"))
    ml_group.add_argument('--nm2', type=file_arg,
                          default=nm2_default,
                          help="Path to pretrained NM2 dtree classifier "
                          "(default - %s)."%(nm2_default))

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
    print(name, rub, kop, file=sys.stderr)    
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
    