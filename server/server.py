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

net, name_recog, digit_recog = None, None, None

def parse_args():
    def file_arg(value): 
        if not path.exists(value):
            if not path.exists(value):
                raise Exception() 
        return value
        
    parser = ArgumentParser()
    
    ml_group = parser.add_argument_group("ML Pretrained Files")
    rub_net_proto_def = path.abspath(path.join(path.dirname(__file__),
                                                "rubli_net/"
                                                "rubli_net.prototxt"))
    ml_group.add_argument('--rub_net_model',
                          default=rub_net_proto_def,
                          help="Path symbol classify net .prototxt "
                          "(default - %s)."%(rub_net_proto_def))
    rub_net_weights_def = path.abspath(path.join(path.dirname(__file__),
                                                  "rubli_net/"
                                                  "pricer_rubli2_iter_100000.caffemodel"))
    ml_group.add_argument('--rub_net_weights',
                          default=rub_net_weights_def,
                          help="Path symbol classify net .caffemodel "
                          "(default - %s)."%(rub_net_weights_def))

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
                          help="Path symbol classify net .prototxt "
                          "(default - %s)."%(name_net_proto_def))
    dig_net_weights_def = path.abspath(path.join(path.dirname(__file__),
                                                 classif_rel_path,
                                                 "price_1_iter_1000.caffemodel"))
    ml_group.add_argument('--digits_net_weights',
                          default=dig_net_weights_def,
                          help="Path symbol classify net .caffemodel "
                          "(default - %s)."%(name_net_proto_def))
    dig_net_dict_def = path.abspath(path.join(path.dirname(__file__),
                                              classif_rel_path,
                                              "price_1_lmdb_dict.json"))
    ml_group.add_argument('--digits_net_dict',
                          default=dig_net_dict_def,
                          help="Path symbol classify net symbols dictionary"
                          "(default - %s)."%(name_net_dict_def))    

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
    
    cnt = net.predict(img)    
    strings, space_idxs, undef_idxs, \
    rects_rub, rects_kop = process_pricer(img, cnt, cnt, cnt)
    name = extract_text_from_rects(img, strings, space_idxs,
                                   undef_idxs, name_recog)
    rub = digit_recog.covert_rects_to_price(img, rects_rub)
    kop = digit_recog.covert_rects_to_price(img, rects_kop)
    print(name, rub, kop, file=sys.stderr)    
    return jsonify({"name": name, "rub": rub,
                    "kop": kop}), 201


if __name__ == '__main__':
    args = parse_args()
    
    net = Localisator(args.rub_net_model, args.rub_net_weights)
    name_recog = SymbolsClassifier(args.name_net_model,
                                       args.name_net_weights,
                                       args.name_net_dict)
    digit_recog = PriceClassifier(args.digits_net_model,
                                  args.digits_net_weights,
                                  args.digits_net_dict)
    
    app.run(debug=True)
    