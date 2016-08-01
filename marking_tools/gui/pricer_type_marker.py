#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:56:07 2016

@author: chernov
"""

import locale
from os import environ, path
import sys

main_dir = path.dirname(path.dirname(path.dirname(__file__)))
if not main_dir in sys.path:
    sys.path.append(main_dir)
print(main_dir)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import marking_tools.markers_db_tools as db_tools
import marking_tools.gui.db_editor as db_editor

DATASET_NAME = "pricer_types"
DB_PATH = 'sqlite:///%s/localization_ML/%s.db'%(environ["BEORGDATA"],
                                               DATASET_NAME)
DATA_DIR = path.join(environ["BEORGDATA"], "localization_ML/images")

if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, 'ru_RU.UTF8')
    
    engine = create_engine(DB_PATH)
    db_tools.Base.metadata.create_all(engine)
    session = (sessionmaker(bind=engine))()
    
    db_editor_ = db_editor.DbEditorImageType(DATA_DIR, session,
                                             resize_size=(600, 800))
    db_editor_.start_marking()
    
