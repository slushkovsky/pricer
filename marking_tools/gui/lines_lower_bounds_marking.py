#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:56:07 2016

@author: chernov
"""

import locale
from os import environ

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from markers_db_tools import Base
from db_editor import DbEditorPoints

DATASET_NAME = "lines_lower_points"
DB_PATH = 'sqlite:///%s/marked_pricers/db/%s.db'%(environ["BEORGDATA"],
                                                  DATASET_NAME)
DATA_DIR = environ["BEORGDATAGEN"] + "/CData_full"

if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, 'ru_RU.UTF8')
    
    engine = create_engine(DB_PATH)
    Base.metadata.create_all(engine)
    session = (sessionmaker(bind=engine))()
    
    db_editor_ = DbEditorPoints(DATA_DIR, session, resize_size=(1200, 600))
    db_editor_.start_marking()
    
