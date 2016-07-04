#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 16:44:23 2016

@author: chernov
"""

import copy
from tkinter.filedialog import Tk
import tkinter.simpledialog
from os import listdir

import cv2

from markers_db_tools import SymbolRel

class DbEditor():
    def __init__(self, data_path, session):
        self.data_path = data_path
        self.session = session
        
        self.root = Tk()
        self.image = None
        self.image_path = None
        self.ix, self.iy = 0, 0
        self.cur_x, self.cur_y = 0, 0
        self.drawing = False
    
    def reload_image(self):
        image_copy = copy.copy(self.image)
        
        symbols = self.session.query(SymbolRel) \
                       .filter(SymbolRel.image==self.image_path)
        for symbol in symbols:
            x1 = int(symbol.x1 * image_copy.shape[1])
            y1 = int(symbol.y1 * image_copy.shape[0])
            x2 = int(symbol.x2 * image_copy.shape[1])
            y2 = int(symbol.y2 * image_copy.shape[0])
            cv2.rectangle(image_copy, 
                          (x1, y1),(x2, y2),
                          (255,0,0), 1)
        
        if self.ix and self.iy and self.cur_x and self.cur_y:
            cv2.rectangle(image_copy, (self.ix, self.iy),
                          (self.cur_x, self.cur_y),(0,255,0), 1)
        cv2.imshow("image", image_copy)
        
    def reset_rect(self):
        self.ix, self.iy = (0, 0)
        self.cur_x, self.cur_y = (0,0) 
    
    def pricer_rect_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x,y
            
        elif event == cv2.EVENT_MBUTTONUP:
            label = tkinter.simpledialog.askstring("11","11")
            
            if not label:
                return
            
            x1 = self.ix / self.image.shape[1]
            y1 = self.iy / self.image.shape[0]
            x2 = self.cur_x / self.image.shape[1]
            y2 = self.cur_y / self.image.shape[0]
            symbol = SymbolRel(x1=x1, y1=y1, x2=x2, y2=y2,
                               label=label, image=self.image_path)
            self.session.add(symbol)
            self.session.commit()
            self.reset_rect()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.reset_rect() 
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                self.cur_x, self.cur_y = x, y
                self.reload_image()
    
    def start_marking(self):
        cv2.namedWindow("image")
        
        def pricer_rect_callback(event, x, y, flags, param):
            return self.pricer_rect_callback(event, x, y, flags, param)
        
        cv2.setMouseCallback("image", pricer_rect_callback)
        
        for file in listdir(self.data_path):
            if file.endswith("jpg"):
                self.image_path = file
                self.image = cv2.imread(self.data_path + "/" + file)
                self.reload_image()
                k = cv2.waitKey() & 0xFF
                if k == 27:
                    cv2.destroyAllWindows()
                    break