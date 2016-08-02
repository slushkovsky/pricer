#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:45:42 2016

@author: chernov
"""
import shutil
from os import path, system, environ, makedirs

BUILD_PATH = path.join(environ["BEORGDATAGEN"], "cv_trained_classifiers/bin")

if(path.exists(BUILD_PATH)):
    shutil.rmtree(BUILD_PATH)
makedirs(BUILD_PATH)

system('cd %s && cmake %s'%(BUILD_PATH, path.abspath(path.dirname(__file__))))
system('cd %s && make'%(BUILD_PATH))