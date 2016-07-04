#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 04:13:39 2016

@author: chernov
"""

import locale
from os import environ

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from markers_db_tools import Base
from markers_db_tools import generate_lmdb

DATASET_NAME = "symbols"
DB_PATH = 'sqlite:///%s/marked_pricers/db/%s.db'%(environ["BEORGDATA"],
                                                  DATASET_NAME)
DATA_DIR = environ["BEORGDATAGEN"] + "/CData_full"

if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, 'ru_RU.UTF8')
    engine = create_engine(DB_PATH)
    Base.metadata.create_all(engine)
    session = (sessionmaker(bind=engine))()
    generate_lmdb(DATA_DIR, session, 
                  DATASET_NAME, show_hists=False,
                  test=False)