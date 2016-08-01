#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 16:59:37 2016

@author: chernov
"""

import random
from datetime import datetime
from os import listdir, makedirs, path, environ
import sys
import shutil
import json

main_dir = path.dirname(__file__)
if not main_dir in sys.path:
    sys.path.append(main_dir)

import cv2
from sqlalchemy import Column, String, DateTime, Integer, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import lmdb
import numpy as np
import caffe

from increase_data import increase_data
from utils.os_utils import show_hist

Base = declarative_base()

class ImageType(Base):
    __tablename__ = 'pricer_types'
    date_added = Column(DateTime, default=datetime.now())
    label = Column(String)
    image = Column(String, primary_key=True)
    def __repr__(self):
        return "<Symbol(added='%s', image='%s', label=%s>" % (
                                self.date_added, self.image, self.label)

class Symbol(Base):
    __tablename__ = 'paths'
    id = Column(Integer, primary_key=True)
    date_added = Column(DateTime, default=datetime.now())
    x1 = Column(Integer)
    y1 = Column(Integer)
    x2 = Column(Integer)
    y2 = Column(Integer)
    label = Column(String)
    image = Column(String)

    def __repr__(self):
        coords = (self.x1, self.y1, self.x2, self.y2)
        return "<Symbol(added='%s', image='%s', label=%s coords ='%s')>" % (
                                self.date_added, self.image, self.label,
                                coords)
        
        
class SymbolRel(Base):
    __tablename__ = 'paths_rel'
    id = Column(Integer, primary_key=True)
    date_added = Column(DateTime, default=datetime.now())
    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    label = Column(String)
    image = Column(String)

    def __repr__(self):
        coords = (self.x1, self.y1, self.x2, self.y2)
        return "<SymbolRel(added='%s', image='%s', label=%s coords ='%s')>" % (
                                self.date_added, self.image, self.label,
                                coords)
        
        
class PointRel(Base):
    __tablename__ = 'points'
    id = Column(Integer, primary_key=True)
    date_added = Column(DateTime, default=datetime.now())
    image = Column(String)
    x = Column(Float)
    y = Column(Float)

    def __repr__(self):
        coords = (self.x, self.y)
        return "<SymbolRel(added='%s', image='%s', coords ='%s')>" % (
                                self.date_added, self.image, coords)
        
        
def get_symbol_image(symbol_rel, image_directory):
    image = cv2.imread(image_directory + "/" + symbol_rel.image)
    shape = image.shape
    x1 = symbol_rel.x1 * shape[1]
    y1 = symbol_rel.y1 * shape[0]
    x2 = symbol_rel.x2 * shape[1]
    y2 = symbol_rel.y2 * shape[0]

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    symbol_image = image[y1:y2, x1:x2]
    return symbol_image
    
    
def get_symbol_image_negative(symbol_rel, image_directory, width=0.3):
    image = cv2.imread(image_directory + "/" + symbol_rel.image)
    shape = image.shape
    x1 = int(symbol_rel.x1 * shape[1])
    y1 = int(symbol_rel.y1 * shape[0])
    x2 = int(symbol_rel.x2 * shape[1])
    y2 = int(symbol_rel.y2 * shape[0])

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
        
    borders = []
    w_half = int(((x2 - x1)*width)//2)
    if x1 - w_half >= 0:
        borders.append(image[y1:y2, x1 - w_half: x1 + w_half])
        
    if x2 + w_half <= shape[1]:
        borders.append(image[y1:y2, x2 - w_half: x2 + w_half])
    return borders
    
    
def view_lmdb(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            
            flat_x = np.fromstring(datum.data, dtype=np.uint8)
            x = flat_x.reshape(datum.channels, datum.width, datum.height)
            y = datum.label
            
            print(y)
            cv2.imshow("image", x.transpose((2, 1, 0)))
            k = cv2.waitKey() & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                return
                
def get_inc_rates(session, increase_iterations=0,
                  merge_upper_and_lower=True,
                  merge_close_symbols=False):
    
    dictionary = session.query(SymbolRel.label).distinct()
    dictionary = [r[0] for r in dictionary]

    if merge_upper_and_lower:
        dictionary = [r.upper() for r in dictionary]
        dictionary = list(set(dictionary))
        
    if merge_close_symbols:
        if 'з' in dictionary:
            dictionary.remove('з')
        if 'З' in dictionary:
            dictionary.remove('З')
        if 'о' in dictionary:
            dictionary.remove('о')
        if 'О' in dictionary:
            dictionary.remove('О')
            
    dict_merge = dict((dictionary[i], set(dictionary[i]))
                      for i in range(0, len(dictionary)))
        
    if merge_upper_and_lower:
        for key in dict_merge.keys():
            dict_merge[key].add(key.lower())
    if merge_close_symbols:
        if '3' in dict_merge:
            dict_merge['3'].add("з")
            dict_merge['3'].add("З")
        if '0' in dict_merge:
            dict_merge['0'].add("о")
            dict_merge['0'].add("О")
            
    labels_list = dictionary
    
    hist = np.zeros(len(dictionary))
    for i in range(0, len(dictionary)):
      hist[i] = session.query(SymbolRel) \
                   .filter(SymbolRel.label.in_(dict_merge[dictionary[i]])).count()
                   
    counts = hist.mean()*3
    
    dictionary = dict((dictionary[i], i) for i in range(0, len(dictionary)))
    
    new_hist= np.zeros(len(dictionary))
    inc_rates = dict()
    
    inc_rates = dict()
    for label in dictionary:
        symbol_counts = hist[dictionary[label]]
        inc_rates[label] = int(counts//symbol_counts)
        new_hist[dictionary[label]] = (hist[dictionary[label]] * 
                                       (1 + inc_rates[label]))
        
    return inc_rates, dictionary, dict_merge, hist, new_hist, labels_list
    
    
def get_label(symb_label, dict_merge):
    label = None
    for key in dict_merge.keys():
        if symb_label in dict_merge[key]:
            label = key
            break
    return label
        
        
def create_session_if_not(dataset_name, session=None):
    if not session:
        db_path = 'sqlite:///%s/marked_pricers/db/%s.db'%(environ["BEORGDATA"],
                                                  dataset_name)
        engine = create_engine(db_path)
        Base.metadata.create_all(engine)
        session = (sessionmaker(bind=engine))()
    return session

    
def generate_lmdb(image_directory, dataset_name, session=None,
                  test_percent = 0.1, 
                  merge_upper_and_lower=True,
                  merge_close_symbols=False,
                  increase_iterations=0, n_chan=3, image_size=(30,30),
                  show_hists=True,
                  test=False):
    session = create_session_if_not(dataset_name, session)
    
    inc_rates, dictionary, dict_merge, \
    hist, new_hist, labels_list = get_inc_rates(session,
                                                increase_iterations,
                                                merge_upper_and_lower,
                                                merge_close_symbols)
    if show_hists:
        show_hist(hist, np.arange(len(labels_list) + 1), labels_list)
        
    min_class = new_hist.min()
    test_samples = dict()
    for label in dictionary:
        samples = random.sample(range(int(new_hist[dictionary[label]])),
                                      int(min_class*test_percent))
        test_samples[label] = samples
    
    map_size = pow(1024,3)
    train_db = lmdb.open('%s/%s_lmdb_train'%(image_directory, dataset_name),
                         map_size=map_size)
    test_db = lmdb.open('%s/%s_lmdb_test'%(image_directory, dataset_name),
                        map_size=map_size)
    symbols = session.query(SymbolRel)
    new_hist_test= np.zeros(len(dictionary))
    j = 0
    k = 0
    cur_hist= np.zeros(len(dictionary))
    with train_db.begin(write=True) as txn_train:
        with test_db.begin(write=True) as txn_test:
            for symbol in symbols:
                label = get_label(symbol.label, dict_merge)
                    
                symbol_image = get_symbol_image(symbol, image_directory)
                art_images = increase_data(symbol_image, inc_rates[label])
                art_images.append(symbol_image)
                
                if test:
                    for i in range(0, len(art_images)):
                        cv2.imshow("%s"%(i), art_images[i])
                    cv2.waitKey()
                    
                for image in art_images:
                    image = cv2.resize(image, image_size, 
                                       interpolation = cv2.INTER_CUBIC)
                    image = image.transpose((2,1,0))
                    shape = image.shape
                    
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.channels = shape[0]
                    datum.width = shape[1]
                    datum.height = shape[2]
                    datum.data = image.tobytes()
                    datum.label = dictionary[label]
                    
                    if (cur_hist[dictionary[label]] in test_samples[label]):
                        str_id = '{:08}'.format(j)
                        txn_test.put(str_id.encode('ascii'),
                                      datum.SerializeToString())
                        new_hist_test[dictionary[label]] += 1
                        j += 1
                    else:
                        str_id = '{:08}'.format(k)
                        txn_train.put(str_id.encode('ascii'),
                                      datum.SerializeToString())
                        k += 1 
                    cur_hist[dictionary[label]]+=1

    label_symbols_dict = dict((dictionary[symb], list(dict_merge[symb]))
                               for symb in dictionary)
    dict_path = '%s/%s_lmdb_dict.json'%(image_directory, dataset_name)
    with open(dict_path, "w") as dict_file:
        dict_file.write(json.dumps(label_symbols_dict))
    
    if show_hists:
        bins = np.arange(len(dictionary) + 1)
        show_hist(new_hist, bins, labels_list)
    
    if show_hists:
        bins = np.arange(len(dictionary) + 1)
        show_hist(new_hist_test, bins, labels_list)
            
    print("labels num: %s"%(len(dictionary)))
    print("train images size: %s"%(k))
    print("test images size: %s"%(j))
    
        
def flush_db(image_directory, dataset_name, session=None, normalise=False):
    session = create_session_if_not(dataset_name, session)
    out_dir = image_directory + "/" + dataset_name
    if(path.exists(out_dir)):
        shutil.rmtree(out_dir)
    makedirs(out_dir)
    
    inc_rates, dictionary, dict_merge = None, None, None
    if normalise:
        inc_rates, dictionary, dict_merge, \
        hist, new_hist, labels_list = get_inc_rates(session, 
                                                    increase_iterations=0,
                                                    merge_upper_and_lower=True,
                                                    merge_close_symbols=True)
    
    symbols = session.query(SymbolRel)
    for symbol in symbols:
        inc_rate = 0
        if normalise:
            inc_rate = inc_rates[get_label(symbol.label, dict_merge)]
                                 
        symbol_image = get_symbol_image(symbol, image_directory)
        images = increase_data(symbol_image, inc_rate)
        images.append(symbol_image)
        
        for i in range(len(images)):
            symbol_out_path = out_dir + "/%s_%s_%s_%s"%(symbol.label,
                                                        symbol.id, i,
                                                        symbol.image)
            cv2.imwrite(symbol_out_path, images[i])
            
            
def flush_db_negative(image_directory, dataset_name, width=0.3,
                      session=None, normalise=False):
    session = create_session_if_not(dataset_name, session)
    out_dir = image_directory + "/%s_negative"%(dataset_name)
    if(path.exists(out_dir)):
        shutil.rmtree(out_dir)
    makedirs(out_dir)
    
    inc_rates, dictionary, dict_merge = None, None, None
    if normalise:
        inc_rates, dictionary, dict_merge, \
        hist, new_hist, labels_list = get_inc_rates(session, 
                                                    increase_iterations=0,
                                                    merge_upper_and_lower=True,
                                                    merge_close_symbols=True)
    
    symbols = session.query(SymbolRel)
    for symbol in symbols:
        inc_rate = 0
        if normalise:
            inc_rate = inc_rates[get_label(symbol.label, dict_merge)]
                                 
        borders = get_symbol_image_negative(symbol, image_directory, width)
        
        for i in range(len(borders)):
            symbol_image = borders[i]
            images = increase_data(symbol_image, inc_rate)
            images.append(symbol_image)
            
            for j in range(len(images)):
                symbol_out_path = out_dir + \
                                 "/%s_%s_%s_%s_%s"%(symbol.label, 
                                                     symbol.id, i, j,
                                                     symbol.image)
                cv2.imwrite(symbol_out_path, images[j])
                

def flush_db_dupl(image_directory, dataset_name,
                  session=None, normalise=False):
    session = create_session_if_not(dataset_name, session)
    out_dir = image_directory + "/%s_negative"%(dataset_name)
    if(path.exists(out_dir)):
        shutil.rmtree(out_dir)
    makedirs(out_dir)
    
    inc_rates, dictionary, dict_merge = None, None, None
    if normalise:
        inc_rates, dictionary, dict_merge, \
        hist, new_hist, labels_list = get_inc_rates(session, 
                                                    increase_iterations=0,
                                                    merge_upper_and_lower=True,
                                                    merge_close_symbols=True)
    
    for file in listdir(image_directory):
        symb_query = session.query(SymbolRel).filter(SymbolRel.image==file).all()
        
        if len(symb_query) <= 1:
            continue

        centers = np.zeros((len(symb_query), 2))
        for i in range(len(symb_query)):
            symb = symb_query[i]
            centers[i] = np.array(((symb.x1 + symb.x2)/2.0,
                                   (symb.y1 + symb.y2)/2.0))
         
        doubles = []
        for i in range(len(symb_query)):
            distances = centers - centers[i]
            distances = ((distances**2).sum(axis=1)**0.5)
            
            min_ = distances.argsort()[1]
            doubles.append((min(i, min_), max(i, min_)))
        
        doubles = list(set(doubles))
        
        for double in doubles:
            image = cv2.imread(path.join(image_directory,file))
            shape = image.shape
            symb1 = symb_query[double[0]]
            symb2 = symb_query[double[1]]

            inc_rate = 0
            if normalise:
                inc_rate = inc_rates[get_label(symb1.label, dict_merge)]
            
            x1 = min(symb1.x1, symb1.x2, symb2.x1, symb2.x2)
            y1 = min(symb1.y1, symb1.y2, symb2.y1, symb2.y2)
            x2 = max(symb1.x1, symb1.x2, symb2.x1, symb2.x2)
            y2 = max(symb1.y1, symb1.y2, symb2.y1, symb2.y2)
            
            x1 *= shape[1]
            x2 *= shape[1]
            y1 *= shape[0]
            y2 *= shape[0]
            
            symbol_image = image[y1:y2, x1:x2].copy()

            images = increase_data(symbol_image, inc_rate)
            images.append(symbol_image)
            
            for j in range(len(images)):
                write_path = path.join(out_dir,
                                       "%s_%s_%s_%s"%(double[0], double[1],
                                                      j, file))
                cv2.imwrite(write_path, images[j])
          

def convert_symbol_to_rel(image_directory, session):
    symbols = session.query(Symbol)
    for symbol in symbols:
        shape = cv2.imread(image_directory + "/" + symbol.image).shape
        x1 = symbol.x1 / shape[1]
        y1 = symbol.y1 / shape[0]
        x2 = symbol.x2 / shape[1]
        y2 = symbol.y2 / shape[0]
        symbol_rel = SymbolRel(x1=x1, y1=y1, x2=x2, y2=y2,
                               label=symbol.label, image=symbol.image)
        session.add(symbol_rel)  
        session.delete(symbol)
    session.commit()
        
        
def correct_db(image_directory, session, dataset_name):
    symbols_dir = image_directory + "/" + dataset_name
    for file in listdir(symbols_dir):
        if file.endswith("jpg"):
            label, id_, image  = file.replace(".jpg", "").split("_")
            symbol = session.query(SymbolRel).filter(SymbolRel.id==id_)
            if symbol.value('label') != label:
                print(("%s: %s -> %s"%(id_, symbol.value('label'), label)))
                symbol.update({SymbolRel.label: label})
                session.commit()

                
if __name__ == "__main__":    
    image_path = path.join(environ["BEORGDATAGEN"], "CData_full")
    flush_db_dupl(image_path, "symbols", normalise=True)
    #flush_db(image_path, "symbols", normalise=True)    
