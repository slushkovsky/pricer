#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:18:08 2016

@author: chernov
"""
import re

import numpy as np
import matplotlib.pyplot as plt

def parse_log(log_path):
    x = np.array(())
    x_loss = np.array(())
    with open(log_path, "r") as log:
        for line in log.readlines():
            matches = re.findall("accuracy = \d+\.\d+", line)
            if len(matches):
                number = float(re.findall("\d+\.\d+", matches[0])[0])
                x = np.append(x, number)
            matches = re.findall("loss = \d+\.\d+", line)
            if len(matches):
                number = float(re.findall("\d+\.\d+", matches[0])[0])
                x_loss = np.append(x_loss, number)
    
    plt.subplot(211)
    plt.plot(np.arange(len(x)), x)
    plt.subplot(212)
    plt.plot(np.arange(len(x_loss)), x_loss)
