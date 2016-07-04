#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 20:43:40 2016

@author: chernov
"""
import sys
from contextlib import contextmanager

@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout


import caffe

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.AdaGradSolver("symbols_net_solver.prototxt")
#solver = caffe.SGDSolver("auto_solver.prototxt")
solver.solve()
