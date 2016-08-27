#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 13:56:42 2016

@author: chernov
"""

import json
from urllib.request import urlopen, Request, HTTPError
import base64
from argparse import ArgumentParser
from os import path, remove, listdir

import cv2

def parse_args():
    def file_arg(value): 
        if not path.exists(value):
            if not path.exists(value):
                raise Exception() 
        return value
        
    parser = ArgumentParser()
    
    parser.add_argument('--image', type=file_arg,
                        help="image path or directory with images")
    parser.add_argument('--uri', default="http://127.0.0.1:5001/price", 
                        help="command uri")
    parser.add_argument('--temp', default="temp.png", 
                        help="temporary image filepath")
    parser.add_argument('--out', default="result.json", 
                        help="file with test results")
    parser.add_argument('--in_json', 
                        help="process failed pricer from json out file")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    req = Request(args.uri)
    req.add_header('Content-Type', 'application/json')

    images = []
    if not args.in_json is None:
        pricers = json.load(open(args.in_json))
        images = [res["file"] for res in pricers if res["code"] != 201]
        print(images)
    elif path.isdir(args.image):
        images = [path.join(args.image, file) for file in listdir(args.image)]
    else:
        images.append(args.image)

    with open(args.out, "w") as file:
        results = []
        try:
            for image in images:
                print("process %s"%(image))
                img = cv2.imread(image, 0)
                if img is None:
                    print("cant load %s as image"%(image))
                    continue
                
                temp_img = args.temp
                cv2.imwrite(temp_img, img)
                
                with open(temp_img, "rb") as f:
                    base64String = base64.b64encode(f.read())
                remove(temp_img)
                    
                message = {"id": 1, "base64String": base64String.decode()}


                result = {}
                result["file"] = image

                try:
                    response = urlopen(req, json.dumps(message).encode("utf-8"))
                    answer = json.loads(response.read().decode("utf-8"))
                    result["code"] = response.code
                    result["answer"] = answer
                    print(answer)
                except HTTPError:
                    print("INTERNAL SERVER ERROR")
                    result["code"] = 500

                results.append(result)
        except KeyboardInterrupt:
            pass
        json.dump(results, file)   
