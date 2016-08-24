#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:13:47 2016

@author: chernov
"""

import glob
import sys
import json
import shutil
from os import path, makedirs
from argparse import ArgumentParser

import cv2
import numpy as np
from scipy.signal import argrelextrema

main_dir = path.abspath(path.join(path.dirname(__file__), "../.."))
if not main_dir in sys.path:
    sys.path.append(main_dir)
del main_dir
cur_dir = path.abspath(path.dirname(__file__))
if not cur_dir in sys.path:
    sys.path.append(cur_dir)
del cur_dir

from classify_symb import PriceClassifier, SymbolsClassifier
from split_symbols_cv import detect_text_cv, normalize_crop, resize_h
from split_price_symbols import process_image    
    
def parse_mark(line, datapath=None, prefix=""):
    """
      Парсинг строки с боксом
      
      @line - тестовая строка с разметкой
      @datapath - путь к папке с избражениями датасета. 
      Если None - изображение не будет загружено
      @prefix - префикс к изображениям в выборке
      @load_img - ффлаг для загрузки изображения
    
      @return - img(оригинальное изображение), cnt(контур бокса)
    """
    s = line.split()
    
    cnt = np.zeros(8, np.float)
    for i in range(1, 9, 2):
        cnt[i - 1] = float(s[i])
        cnt[i] = float(s[i + 1])
    cnt = cnt.reshape((4,2)).astype(np.int)
    
    if not datapath is None:
        img = cv2.imread(path.join(datapath, prefix + s[0]))
        return img, cnt
    else:
        return cnt
    
    
def prepare_mark(datapath, line, prefix="", new_h=700):
    """
      Подготовка ценника
      Функция парсит разметку и пережимает изображение и контуры под заданный
      размер
      
      @datapath - путь к папке с избражениями датасета
      @line - тестовая строка с разметкой
      @prefix - префикс к изображениям в выборке
      @new_h - высота выходного изображения
    
      @return - img(изображение), cnt(контур бокса), scale(коэф. масштабир.)      
    """
    img, cnt = parse_mark(line, datapath, prefix=prefix)
    if img is None:
        return None, cnt
    img, scale = resize_h(img, new_h=new_h)
    cnt = (cnt * scale).astype(np.int)
    return img, cnt, scale
     
    
def open_pricer(datapath, line, prefix="", new_h=700):
    """
      Подготовка ценника
      Функция парсит разметку и пережимает изображение и контуры под заданный
      размер
      
      @datapath - путь к папке с избражениями датасета
      @line - тестовая строка с разметкой
      @prefix - префикс к изображениям в выборке
      @new_h - высота выходного изображения
    
      @return - img(изображение), cnt(контур бокса), scale(коэф. масштабир.).
                В случае не успешной обработки - None, None, None      
    """
    try:
        img, cnt, scale = prepare_mark(datapath,
                                         line,
                                         new_h=new_h)
        return img, cnt, scale
    except ValueError:
        print("cant process image %s"%(path.join(datapath, key)))
        return None, None, None
    
    
def extract_symbol_rects(crop, nm1, nm2, offset=(0,0), min_variance=200):
    """
      Выделение боксов символов из названия ценника
      
      @crop - обрезанное название
      @nm1 - путь к файлу NM1 классификатора
      @nm1 - путь к файлу NM2 классификатора
      @offset - отступ обрезанного названия
      @min_variance - минимальная вариация в символе
    
      @return - img(изображение), cnt(контур бокса), scale(коэф. масштабир.).
                В случае не успешной обработки - None, None, None      
    """
    
    rects, rects_bad = [], []
    regions = [[0,  crop.shape[0]]]#crop_regions(split_lines_hist(crop), crop.shape[0]*0.15)   
    for i in range(len(regions)):
        cur_img_part = crop[regions[i][0]:regions[i][1],:]
        cur_rects, cur_rects_bad = detect_text_cv(cur_img_part,
                                                  nm1, 
                                                  nm2,
                                                  min_variance=min_variance)
        for rect in cur_rects:
            rect[1] += regions[i][0] + offset[1]
            rect[0] += offset[0]
            rects.append(rect)
            
        for rect in cur_rects_bad:
            rect[1] += regions[i][0] + offset[1]
            rect[0] += offset[0]
            rects_bad.append(rect)
    return rects, rects_bad
    
     
def process_name(img, cnt, nm1, nm2, min_var=0, min_symb_var=0):    
    if img is None:
        raise Exception("image is None")
        
    name_rect = cv2.boundingRect(cnt)
    crop = img[name_rect[1]:name_rect[1]+name_rect[3],
               name_rect[0]:name_rect[0]+name_rect[2]]

    #img[name_rect[1]:name_rect[1]+name_rect[3],
    #    name_rect[0]:name_rect[0]+name_rect[2]] = normalize_crop(crop)    

    if min_var > 0:
        var = variance_of_laplacian(crop)
        if args.img:
            print("variance - %s"%(var))
        if var < args.min_var:
            raise Exception("too blured (%s < %s)"%(var, min_var))
    try:
        rects, rects_bad = extract_symbol_rects(crop, nm1, nm2,
                                                offset=(name_rect[0],
                                                        name_rect[1]), 
                                                min_variance=min_symb_var)
        return rects, name_rect
        
    except ValueError:
        print("cant extract name symbols")
        return [], name_rect
    
        

def process_price(img, cnt):
    if img is None:
        raise Exception("image is None")
    
    price_rect = cv2.boundingRect(cnt)
    crop = img[price_rect[1]:price_rect[1]+price_rect[3],
               price_rect[0]:price_rect[0]+price_rect[2]]

    try:
        rects_price = extract_price_rects(crop, offset=(price_rect[0],
                                                        price_rect[1]))
        rects_price = sorted(rects_price, key=lambda x: x[0])
        return rects_price, price_rect
        
    except ValueError:
        print("cant extract price symbols")
        return [], price_rect


def calc_rect_var_gap_thrs(img, rects, w_perc):
    """
      Вычисление вариации и среднего значения для промежутков между символами
    
      Алгоритм:
          - Вычисление вариации всех символов
          - Фильтрация w_perc процентов символов по отклонению 
          от средней вариации
          - Вычисление среднего и вариации для отфильтрованнных символов
      
      @rects - Боксы символов
      @w_perc - Процент лучших промежутков, по которым будут вычисляться
      значения
      @return - Среднее значение и вариация промежутков между символами
    """
    variances = np.zeros((len(rects)))
        
    for i, symbol in enumerate(rects):
        symb_img = img[symbol[1]: symbol[1] + symbol[3],
                       symbol[0]: symbol[0] + symbol[2]]
        variances[i] += symb_img.var()
   
    mean_var = np.mean(variances)
    
    symb_vars_disp = np.sort(np.abs(variances[:] - mean_var))
    var_thr = symb_vars_disp[int(len(symb_vars_disp)*w_perc)]  
    
    mean_var = variances[symb_vars_disp < var_thr].mean()
    std_var = variances[symb_vars_disp < var_thr].std()
    return mean_var, std_var
    
    
def get_rect_text_lines(img, rects, conv_symb_h_ratio=2):
    """
      Выделение горизонтальных линий, соответсвующих строкам в тексте.
      
      @img - Изображение
      @rects - Боксы символов
      @conv_symb_h_ratio - Размер ядра свертки при сглаживании гистограммы
      @return - Лист y-координат линий
      @rparam - numpy.array
    """
    
    symbols = np.zeros(img.shape[0:2], np.uint8)
    for rect in rects:
        symbols[rect[1]: rect[1] + rect[3],
                rect[0]: rect[0] + rect[2]] = 255

    hist = symbols.sum(axis=1)
    mean_symb_h = int(np.mean(rects, axis=0)[3])
    hist = np.convolve(hist, np.full((mean_symb_h//conv_symb_h_ratio),
                                     1, np.int64),'same')
    lines = argrelextrema(hist, np.greater_equal,
                          order=mean_symb_h//conv_symb_h_ratio)[0]
    lines = lines[hist[lines]>0]

    #сливание близких строк
    lines = [line for i, line in enumerate(lines) \
             if i == len(lines) - 1 or not\
             (hist[line] == hist[lines[i + 1]] and
              lines[i + 1] - line < mean_symb_h)]
    lines = np.sort(lines)
    return lines
    
    
def split_rects_by_strings(img, rects, conv_symb_h_ratio=2):
    
    """
      Разделение символов текста на строки
      
      @img - Изображение
      @rects - Боксы символов
      @conv_symb_h_ratio - Размер ядра свертки при сглаживании гистограммы
      @return - Отсортированнык боксы, разделенные по строкам
      @rparam - list(list())
    """
    
    if len(rects) == 0:
        return []
        
    mean_symb_h = int(np.mean(rects, axis=0)[3])
    lines = get_rect_text_lines(img, rects, conv_symb_h_ratio)
    
    strings = []
    for line in lines:
        rects_line = [rect for rect in rects \
                      if abs((rect[1] + rect[3]/2) - line) < mean_symb_h]
        rects_line = sorted(rects_line, key=lambda x: x[0])
        strings.append(rects_line)
    return strings
    
    
def filter_rect_string(img, string, w_perc=0.8, std_perc=0.8,
                       gap_mean_w_ratio=0.2, space_at_end=False):
    """
      Фильтрация боксов на строке
      При фильтрации происходит детектирование пробелов и незахваченных
      символов.
      
      Алгоритм:
          1. Вычисление гистограммы расстояний между символами.
          2. Оценка примерного количества пробелов на строке
          3. Определение порога по оценке количества пробелов и гистограмме
          4. Определение порога вариации по w_perc лучших символов
          5. Порог по вариации умножается на std_perc
          6. Боксы последовательно добавляются в выходной лист. Если
             расстояние между боксами больше порога - проводится проверка
             зазора между боксом и его классификация(пробел, символ)
      
      @img - изображение
      @string - боксы на строке
      @w_perc - Процент лучших символов, по которым будет определяться порог
      вариации
      @std_perc - множитель для порога по вариации
      @gap_mean_w_ratio - минимально допустимая ширина пробела относительно
      средней ширины символа
      @space_at_end - добавление в конец строки пробела
    """
    
    distances = np.array(())
    for i in range(len(string) - 1):
        x1 = max(string[i][0], string[i][0] + string[i][2])
        x2 = min(string[i + 1][0], string[i + 1][0] + \
                 string[i + 1][2])
        distances = np.append(distances, x2 - x1)
        
    hist, bins = np.histogram(distances, 8)
    #show_hist(hist, bins)
    hist = np.cumsum(hist[::-1])[::-1]
    #show_hist(hist, bins)
    
    words_approx = len(string) / 7.2
    gap_thr = np.argmax(hist < words_approx)
    idx = np.argmax(hist == hist[gap_thr])
    if bins[idx - 1] - bins[idx - 1] < words_approx / 3.0:
        idx -= 1
    gap_thr = bins[idx]

    mean_symb_width = int(np.mean(string, axis=0)[2])
    gap_thr = max(int(mean_symb_width*gap_mean_w_ratio), gap_thr)
    
    #print(gap_thr)
    #show_hist(distances, np.arange(len(distances) + 1))
    
    mean_var, std_var = calc_rect_var_gap_thrs(img, string, w_perc=w_perc)
    
    line_rects, undef_idx, space_idx = [], [], []
    for i, symbol in enumerate(string):
        line_rects.append(symbol)
        if i + 1 < len(string):
            if distances[i] > gap_thr:
                symb2 = string[i + 1]
                x0 = max(symbol[0] + symbol[2], symbol[0])
                y0 = min(symbol[1], symb2[1])
                x1 = min(symb2[0] + symb2[2], symb2[0])
                y1 = max(symb2[1], symb2[1] + symb2[3])
                
                del_rect = [x0, y0, x1 - x0, y1 - y0]
                del_img = img[del_rect[1]: del_rect[1] + del_rect[3],
                              del_rect[0]: del_rect[0] + del_rect[2]]
                line_rects.append(del_rect)
                if del_img.var() < mean_var + std_var*std_perc:
                    space_idx.append(len(line_rects) - 1)
                else:
                    undef_idx.append(len(line_rects) - 1)
        elif space_at_end:
            del_rect = [symbol[0] + symbol[2], symbol[1], 
                        symbol[3], symbol[3]]
            line_rects.append(del_rect)
            space_idx.append(len(line_rects) - 1)
                    
    return line_rects, undef_idx, space_idx


def extract_price_rects(crop, offset=(0,0)):
    rects = process_image(crop)
    for i in range(len(rects)):
        rects[i][0] += offset[0]
        rects[i][1] += offset[1]
    return rects


def merge_mark_files(files):
    """
      Объединение нескольких файлов разметки в один
      
      @files - Список путей к файлам разметки
      @return - Карта разметки с названиями файлов в качестве ключей 
      @rparam - set
    """
    markings = dict()
    for file in files:
        for line in open(file, "r").readlines():
            markings[line.split()[0]] = line
    return markings


def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()    


def parse_args():
    def file_arg(value): 
        if not path.exists(value):
            if not path.exists(value):
                raise Exception() 
        return value
        
    nets_path = path.join(path.abspath(path.join(__file__, "../../..")),
                          "ml_data")
    
    class_dig_path = path.join(nets_path, "class_digits")
    class_dig_proto_def = path.join(class_dig_path, "digits_net.prototxt")
    class_dig_weights_def = path.join(class_dig_path, "digits_net.caffemodel")
    class_dig_dict_def = path.join(class_dig_path, "digits_net_dict.json")
    
    class_symb_path = path.join(nets_path, "class_symb")
    class_symb_proto_def = path.join(class_symb_path, "symbols_net.prototxt")
    class_symb_weights_def = path.join(class_symb_path, "symbols_net.caffemodel")
    class_symb_dict_def = path.join(class_symb_path, "symbols_net_dict.json")
    
    nm_path = path.join(nets_path, "NM")
    nm1_def = path.join(nm_path,"trained_classifierNM1.xml")
    nm2_def = path.join(nm_path,"trained_classifierNM1.xml")    
        
    parser = ArgumentParser()
    
    data_group = parser.add_argument_group("Data Settings")
    data_group.add_argument('datapath', type=file_arg,
                            help="Path to folder with image data.")
    data_group.add_argument('names_wildcard', type=str,
                            help="Wildcard expression with names marks files.")
    data_group.add_argument('rubli_wildcard', type=str,
                            help="Wildcard expression with rubles marks files.")
    data_group.add_argument('kopeck_wildcard', type=str,
                            help="Wildcard expression with kopecks marks files.")
    data_group.add_argument("--img", help="Execute for single image in dataset.")
    
    params_group = parser.add_argument_group("Algoritm Parameters")
    params_group.add_argument("--resize_h", type=int, default=700,
                               help="Internal processing pricer height size "
                               "(default - 700).")
    params_group.add_argument("--min_var", type=int, default=-1,
                              help="Minimal variance of pricer name field. -1 "
                              "if disabled (default - (-1)).")
    params_group.add_argument("--min_symb_var", type=int, default=200,
                              help="Minimal variance of pricer (default 200).")
    
    ml_group = parser.add_argument_group("ML Pretrained Files")
    ml_group.add_argument('--nm1', type=file_arg, default=nm1_def,
                          help="Path to pretrained NM1 dtree classifier "
                          "(default - %s)."%(nm1_def))
    ml_group.add_argument('--nm2', type=file_arg, default=nm2_def,
                          help="Path to pretrained NM2 dtree classifier "
                          "(default - %s)."%(nm2_def))
    
    ml_group.add_argument('--name_net_model', default=class_symb_proto_def,
                          help="Path symbol classify net .prototxt "
                          "(default - %s)."%(class_symb_proto_def))
    ml_group.add_argument('--name_net_weights', default=class_symb_weights_def,
                          help="Path symbol classify net .caffemodel "
                          "(default - %s)."%(class_symb_weights_def))
    ml_group.add_argument('--name_net_dict', default=class_symb_dict_def,
                          help="Path symbol classify net symbols dictionary"
                          "(default - %s)."%(class_symb_dict_def))
    
    ml_group.add_argument('--digits_net_model', default=class_dig_proto_def,
                          help="Path symbol classify net .prototxt "
                          "(default - %s)."%(class_dig_proto_def))
    ml_group.add_argument('--digits_net_weights', default=class_dig_weights_def,
                          help="Path symbol classify net .caffemodel "
                          "(default - %s)."%(class_dig_weights_def))
    ml_group.add_argument('--digits_net_dict', default=class_dig_dict_def,
                          help="Path symbol classify net symbols dictionary"
                          "(default - %s)."%(class_dig_dict_def))
    
    output_group = parser.add_argument_group("Output Settings")
    out_def = path.join(path.dirname(__file__),"split_data")
    output_group.add_argument("--outdir", default=out_def,
                              help="Output directory (default - %s)."%(out_def))
    output_group.add_argument("--out_vis", action="store_true",
                               help="Save images with marks.")
    output_group.add_argument("--resize_out", action="store_true",
                               help="Resize output image. If enabled, resised "
                               "height will be equal --resize_h value.")
    output_group.add_argument("--out_box", action="store_true",
                              help="Save symbol markings to .box file")
    output_group.add_argument("--out_json", action="store_true",
                              help="Save output to json file. This flag will "
                              "activate symbol classification step. Proper "
                              "name net parameters (--name_net_dict, "
                              "--name_net_weights, --name_net_model) is needed")
    return parser.parse_args()
    
    
def process_pricer(img, nm1, nm2, name_сnt=None, rub_сnt=None, kop_сnt=None,
                   min_var=0, min_symb_var=0):
    """
      Выделение символов из ценника
      
      @img - изображение ценника
      @name_сnt - контур цены (4 вершины в формате cv2)
      @rub_сnt - контур рублей (4 вершины в формате cv2)
      @kop_сnt - контур копеек (4 вершины в формате cv2)
      @min_var - минимальная вариация в названии
      
      @return - strings - упорядоченный лист символов названия
                space_idxs - индексы пробелов в символах названия
                undef_idxs - индексы пропущенных символов названия
                rects_rub - упорядоченный лист символов рублей
                rects_kop - упорядоченный лист символов копеек
                
    """
    space_idxs, undef_idxs = [], []
    strings, rects_rub, rects_kop = [], [], []
    name_rect, rub_rect, kop_rect = None, None, None

    if not name_сnt is None:
        rects, name_rect = process_name(img, name_сnt, nm1=nm1, nm2=nm2,
                                        min_var=min_var,
                                        min_symb_var=min_symb_var)
        strings = split_rects_by_strings(img, rects)
        # фильтрация прямоугольников
        str_all = []
        for i in range(len(strings)):
            string, undef_idx, space_idx = filter_rect_string(img, strings[i])
            undef_idxs += list(np.array(undef_idx) + len(str_all))
            space_idxs += list(np.array(space_idx) + len(str_all))
            str_all += string
        strings = str_all
        
    if not rub_сnt is None:
        rects_rub, rub_rect = process_price(img, rub_сnt)
        
    if not kop_сnt is None:
        rects_kop, kop_rect = process_price(img, kop_сnt)
        
    return strings, space_idxs, undef_idxs, rects_rub, rects_kop
    
    
def extract_text_from_rects(img, strings, space_idxs, undef_idxs,  name_recog):
    """
      Конвертация боксов названия в текст
      
      @img - изображение
      @strings - урорядоченный лист боксов с пробелами
      @space_idxs - индексы пробелов
      @undef_idxs - индексы нелокализованных символов
      @name_recog - классификатор
      @return - распознанный текст
      @rtype - str
    """   
    name = np.array(["0" for i in range(len(strings))])
    if len(space_idxs): name[space_idxs] = " "
    if len(undef_idxs): name[undef_idxs] = "*"
    for i in range(name.shape[0]):
        if name[i] == "0":
            symbol = strings[i]
            name[i] = name_recog.predict(img[symbol[1]: 
                                             symbol[1] + symbol[3],  
                                             symbol[0]:  
                                             symbol[0] + symbol[2]])[0]
    return "".join(name)

    
if __name__ == "__main__":
    args = parse_args()
    
    names_marks = merge_mark_files(glob.glob(args.names_wildcard))
    rubles_marks = merge_mark_files(glob.glob(args.rubli_wildcard))
    kopecks_marks = merge_mark_files(glob.glob(args.kopeck_wildcard))
    
    keys = []
    keys += list(names_marks.keys())
    keys +=list(rubles_marks.keys())
    keys +=list(kopecks_marks.keys())
    keys = list(set(keys))
    
    name_recog = None
    digit_recog = None
    if args.out_json:
        name_recog = SymbolsClassifier(args.name_net_model,
                                       args.name_net_weights,
                                       args.name_net_dict)
        digit_recog = PriceClassifier(args.digits_net_model,
                                      args.digits_net_weights,
                                      args.digits_net_dict)
    
    batch = []
    if args.img:
        batch.append(args.img)
    else:
        batch += keys
        
    if args.out_box or args.out_json or args.out_vis:
        if path.exists(args.outdir):
            shutil.rmtree(args.outdir)
        makedirs(args.outdir)
    
    for key in batch:
        
        if not path.exists(path.join(args.datapath, key)):
            print("image %s not found"%(path.join(args.datapath, key)))
            continue
        
        scale, img = 1, None 
        name_cnt, rub_cnt, kop_cnt = None, None, None
        
        if key in names_marks:
            img, name_cnt = parse_mark(names_marks[key], args.datapath)
        else:
            continue
        
        if key in rubles_marks:
            rub_cnt = parse_mark(rubles_marks[key], None)
            
        if key in kopecks_marks:
            kop_cnt = parse_mark(kopecks_marks[key], None)
        
        strings, space_idxs, undef_idxs = None, None, None
        rects_rub, rects_kop = None, None
        
        try:
            strings, space_idxs, undef_idxs, \
            rects_rub, rects_kop = process_pricer(img, args.nm1, args.nm2,
                                                  name_cnt, rub_cnt,
                                                  kop_cnt, min_var=args.min_var,
                                                  min_symb_var=args.min_symb_var)
        except Exception as e: 
                print("%s Exception: %s"%(key, e))
                continue  
            
        vis = img
        if args.resize_out:
            vis, scale = resize_h(img, args.resize_h)
            img = vis
            rects_kop = [(np.array(r)*scale).astype(np.int) for r in rects_kop]
            rects_rub = [(np.array(r)*scale).astype(np.int) for r in rects_rub]
            for i in range(len(strings)):
                strings[i] = [(np.array(r)*scale).astype(np.int) for r \
                              in strings[i]]

        # объединение отфильтрованных символов
        if args.out_box:
            out = open(path.join(args.outdir, 
                                 path.splitext(path.basename(key))[0] + 
                                 ".box"), "w")
            batch = {"n": strings,
                     "r": rects_rub, "k": rects_kop}
            for ele in batch:
                for rect in batch[ele]:
                    out.write("* %s %s %s %s 0 %s\n"%(int(rect[0]), 
                                                     int(rect[1] + rect[3]), 
                                                     int(rect[0] + rect[2]), 
                                                     int(rect[1]), ele))
            out.close()
            
        if args.out_json:
            dict_ = {"name": "", "rub": -1, "kop": -1}

            dict_["name"] = extract_text_from_rects(img, strings, space_idxs,
                                                    undef_idxs, name_recog)
            if len(rects_rub) > 0:
                dict_["rub"] = digit_recog.covert_rects_to_price(img, 
                                                                 rects_rub)
            if len(rects_kop) > 0:
                dict_["kop"] = digit_recog.covert_rects_to_price(img, 
                                                                 rects_kop)
                
            json.dump(dict_, open(path.join(args.outdir,
                                            path.splitext(path.basename(key))[0] +
                                  ".json"), "w"),
                      ensure_ascii=False)
                
        if args.out_vis:
            #for rect in [name_rect, rub_rect, kop_rect]:
            #    if rect:
            #        cv2.rectangle(vis, (rect[0], rect[1]), 
            #                      (rect[0] + rect[2], 
            #                       rect[1] + rect[3]), 
            #                      (255, 0, 0), 1)
            
            spaces = np.array(strings)[space_idxs]
            undefs = np.array(strings)[undef_idxs]
            symbols = np.delete(strings,
                                np.append(space_idxs, undef_idxs), axis=0)
            
            for rects, color in [[symbols, (255,0,0)], [spaces, (0,255,255)],
                                 [undefs, (255, 0,255)]]:
                for rect in rects:
                    cv2.rectangle(vis, (rect[0], rect[1]),
                                  (rect[0] + rect[2], rect[1] + rect[3]),
                                  color, 1)
                              
            for rects in [rects_kop, rects_rub]:
                for rect in rects:
                    cv2.rectangle(vis, (rect[0], rect[1]),
                                  (rect[0] + rect[2], rect[1] + rect[3]),
                                  (0, 255, 0), 1)
            cv2.imwrite(path.join(args.outdir, key), vis)
            
        if args.img:
            cv2.imshow("vis", vis)
            cv2.waitKey()
            cv2.destroyAllWindows()
            