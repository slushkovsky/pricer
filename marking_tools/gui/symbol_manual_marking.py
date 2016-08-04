#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 00:14:43 2016

@author: chernov
"""

import locale
from os import environ, path
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

tools_dir = path.dirname(path.dirname(path.realpath(__file__)))
if not tools_dir in sys.path:
    sys.path.append(tools_dir)
del tools_dir

from markers_db_tools import Base
from db_editor import DbEditor

DATASET_NAME = "nochar"
DB_PATH = 'sqlite:///%s/marked_pricers/db/%s.db'%(environ["BEORGDATA"],
                                                  DATASET_NAME)
DATA_DIR = environ["BEORGDATAGEN"] + "/CData_full"

if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, 'ru_RU.UTF8')
    
    engine = create_engine(DB_PATH)
    Base.metadata.create_all(engine)
    session = (sessionmaker(bind=engine))() 
    
    db_editor_ = DbEditor(DATA_DIR, session, resize_size=(800, 400))
    db_editor_.start_marking()

