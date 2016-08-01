#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 16:44:23 2016

@author: chernov
"""

import copy
from tkinter.filedialog import Tk
import tkinter.simpledialog
from os import listdir, path
import sys

import cv2
from sqlalchemy import and_

tools_dir = path.dirname(path.dirname(path.realpath(__file__)))
if not tools_dir in sys.path:
    sys.path.append(tools_dir)
del tools_dir

import markers_db_tools

ImageType = markers_db_tools.ImageType
SymbolRel = markers_db_tools.SymbolRel
PointRel = markers_db_tools.PointRel

class DbEditorImageType():
    """ Gui интерфейс для разметки типа изображений
    
    Управление:
       q - отметить ценник с контрастным фоном
       w - отметить ценник с неконтрастным фоном
       e - следующее изображение
       r - предыдущее изображение
    """
    
    def __init__(self, data_path, session, 
                 resize_size=None):
        self.data_path = data_path
        self.session = session
        self.image = None
        self.image_path = None
        self.resize_size = resize_size
        self.i = 0
        
    def reload_image(self):
        image_copy = copy.copy(self.image)
        types = self.session.query(ImageType) \
                     .filter(ImageType.image==self.image_path)
        for type_ in types:
            cv2.putText(image_copy, type_.label,
                        (0, image_copy.shape[0]//2),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 4,
                        (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("image", image_copy)
        
    def update_label(self, label):
        type_ = ImageType(label=label, image=self.image_path)
        self.session.merge(type_)
        self.session.commit()
        self.reload_image()
    
    def start_marking(self):
        cv2.namedWindow("image")
        
        files = [f for f in listdir(self.data_path) if f.endswith('.jpg')]
        while self.i < len(files):
            self.image_path = files[self.i]
            self.image = cv2.imread(path.join(self.data_path, files[self.i]))
            if self.resize_size:
                self.image = cv2.resize(self.image, self.resize_size,
                                        cv2.INTER_CUBIC)
            self.reload_image()
            k = cv2.waitKey() 
            if k & 0xFF == 27:
                cv2.destroyAllWindows()
                break
            elif k == ord('q') or  k == ord('Q'):
                self.update_label("contrast")
            elif k == ord('w') or  k == ord('W'):
                self.update_label("nocontrast")
            elif k == ord('e') or  k == ord('E'):
                self.i += 1
            elif k == ord('r') or  k == ord('R'):
                self.i = max(0, self.i - 1)
                
                    
        cv2.destroyAllWindows()

    
class DbEditorPoints():
    def __init__(self, data_path, session, 
                 resize_size=None, remove_error=0.01):
        self.data_path = data_path
        self.session = session
        self.image = None
        self.image_path = None
        self.remove_error = remove_error
        self.resize_size = resize_size
        
    def reload_image(self):
        image_copy = copy.copy(self.image)
        points = self.session.query(PointRel) \
                     .filter(PointRel.image==self.image_path)
        for point in points:
            x = int(point.x * image_copy.shape[1])
            y = int(point.y * image_copy.shape[0])
            cv2.circle(image_copy, (x, y), 1, (0,0,255), 2)
            cv2.circle(image_copy, (x, y), 5, (0,0,255), 1)
        cv2.imshow("image", image_copy)
        
    def pricer_rect_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            
            point = PointRel(x=x/self.image.shape[1],
                             y=y/self.image.shape[0],
                             image=self.image_path)
            self.session.add(point)
            self.session.commit()
            self.reload_image()
            
        if event == cv2.EVENT_RBUTTONDBLCLK:
            x_upper = x/ self.image.shape[1] + self.remove_error
            x_lower = x/ self.image.shape[1] - self.remove_error
            y_upper = y/ self.image.shape[0] + self.remove_error
            y_lower = y/ self.image.shape[0] - self.remove_error
            
            points = self.session.query(PointRel) \
                         .filter(and_(PointRel.image==self.image_path,
                                      PointRel.x >= x_lower,
                                      PointRel.x <= x_upper,
                                      PointRel.y >= y_lower,
                                      PointRel.y <= y_upper))
            for point in points:
                self.session.delete(point)
            self.session.commit()
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
                if self.resize_size:
                    self.image = cv2.resize(self.image, self.resize_size,
                                            cv2.INTER_CUBIC)
                self.reload_image()
                k = cv2.waitKey() & 0xFF
                if k == 27:
                    cv2.destroyAllWindows()
                    break
        cv2.destroyAllWindows()
    

class DbEditor():
    def __init__(self, data_path, session, resize_size=None):
        self.data_path = data_path
        self.session = session
        
        self.root = Tk()
        self.image = None
        self.image_path = None
        self.ix, self.iy = 0, 0
        self.cur_x, self.cur_y = 0, 0
        self.drawing = False
        self.resize_size = resize_size
    
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
                if self.resize_size:
                    self.image = cv2.resize(self.image, self.resize_size,
                                            cv2.INTER_CUBIC)
                self.reload_image()
                k = cv2.waitKey() & 0xFF
                if k == 27:
                    cv2.destroyAllWindows()
                    break
        cv2.destroyAllWindows()