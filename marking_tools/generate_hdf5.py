#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 21:42:01 2016

@author: chernov
"""

from os import environ
from argparse import ArgumentParser

from dataset_utils import generate_h5_db


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dataset_name')
    return parser.parse_args()

    
if __name__ == '__main__':
    args = parse_args()
    
    dataset_name = args.dataset_name
    marking_path = '%s/marked_pricers/%s.txt'%(environ["BEORGDATA"], dataset_name)
    data_dir = environ["BEORGDATAGEN"] + "/CData_full"
    
    db_name_train = environ["BEORGDATAGEN"] + \
                    "/CData_full/train_%s.h5"%(dataset_name)
                    
    db_name_test = environ["BEORGDATAGEN"] + \
                   "/CData_full/test_%s.h5"%(dataset_name)
    
    generate_h5_db(marking_path, data_dir, 
                   db_name_train, db_name_test, image_size=(30,15),
                   original_image_size=(500, 250))