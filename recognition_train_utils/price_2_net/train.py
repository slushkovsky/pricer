#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 20:43:40 2016

@author: chernov
"""

import caffe

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.AdaGradSolver("price_2_net_solver.prototxt")
solver.solve()
