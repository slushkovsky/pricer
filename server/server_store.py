#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:19:13 2016

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

from beorg_net_utils import Classificator

store_recog = None
args = None

def parse_args():
    def file_arg(value): 
        if not path.exists(value):
            if not path.exists(value):
                raise Exception() 
        return value
        
    nets_path = path.join(path.abspath(path.join(__file__, "../..")),
                          "ml_data")
    
    class_store_path = path.join(nets_path, "class_store")
    class_store_proto_def = path.join(class_store_path, "store_net.prototxt")
    class_store_weights_def = path.join(class_store_path, "store_net.caffemodel")
    class_store_dict_def = path.join(class_store_path, "store_net_dict.json")
    
    parser = ArgumentParser()
    
    ml_group = parser.add_argument_group("ML Pretrained Files")
    
    ml_group.add_argument('--store_net_model', default=class_store_proto_def,
                          help="Path store classify net .prototxt "
                          "(default - %s)."%(class_store_proto_def))
    ml_group.add_argument('--store_net_weights', default=class_store_weights_def,
                          help="Path store classify net .caffemodel "
                          "(default - %s)."%(class_store_weights_def))
    ml_group.add_argument('--store_net_dict', default=class_store_dict_def,
                          help="Path store classify net symbols dictionary"
                          "(default - %s)."%(class_store_dict_def))

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
    
    store_id = store_recog.predict(img)[0]
    return jsonify({"shopId": store_id, "id": request.json["id"]}), 201


if __name__ == '__main__':
    args = parse_args()
    
    store_recog = Classificator(args.store_net_model,
                                args.store_net_weights,
                                args.store_net_dict,
                                out_layer="prob")
    app.run(host='0.0.0.0', debug=False)