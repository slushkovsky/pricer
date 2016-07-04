#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 04:13:39 2016

@author: chernov
"""

import locale
from os import environ
from argparse import ArgumentParser

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from markers_db_tools import Base
from markers_db_tools import generate_lmdb



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dataset_name')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    dataset_name = args.dataset_name
    db_path = 'sqlite:///%s/marked_pricers/db/%s.db'%(environ["BEORGDATA"],
                                                      dataset_name)
    data_dir = environ["BEORGDATAGEN"] + "/CData_full"
    
    locale.setlocale(locale.LC_ALL, 'ru_RU.UTF8')
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    session = (sessionmaker(bind=engine))()
    generate_lmdb(data_dir, session, 
                  dataset_name, show_hists=False,
                  test=False)