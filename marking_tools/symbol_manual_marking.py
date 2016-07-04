#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 00:14:43 2016

@author: chernov
"""

import locale
from os import environ

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from markers_db_tools import Base
from db_editor import DbEditor
from markers_db_tools import flush_db, correct_db, generate_lmdb
from markers_db_tools import convert_symbol_to_rel, view_lmdb

DATASET_NAME = "symbols"
#DATASET_NAME = "price_1"
#DATASET_NAME = "price_2"

DB_PATH = 'sqlite:///%s/marked_pricers/db/%s.db'%(environ["BEORGDATA"],
                                                  DATASET_NAME)
DATA_DIR = environ["BEORGDATAGEN"] + "/CData_full"

if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, 'ru_RU.UTF8')
    
    engine = create_engine(DB_PATH)
    Base.metadata.create_all(engine)
    session = (sessionmaker(bind=engine))()
    
    #db_editor_ = DbEditor(DATA_DIR, session)
    #db_editor_.start_marking()
    
    #view_lmdb(DATA_DIR + "/" + DATASET_NAME +"_lmdb_test")
    #convert_symbol_to_rel(DATA_DIR, session)
    generate_lmdb(DATA_DIR, session, DATASET_NAME)
    #flush_db(DATA_DIR, session, DATASET_NAME)
    #correct_db(DATA_DIR, session, DATASET_NAME)

