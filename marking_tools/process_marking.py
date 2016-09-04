#!/usr/bin/ python
# -*- coding: utf-8 -*-

import sys
import shutil
from collections import namedtuple
from argparse import ArgumentParser
from os import listdir, path, makedirs

import cv2

main_dir = path.abspath(path.join(path.dirname(__file__), ".."))
if not main_dir in sys.path:
    sys.path.append(main_dir)
del main_dir

from symbol_split_utils.cv_standalone.split_symbols_cv import resize_h

DICT = {
    'А': 'А',
    'Б': 'Б',
    'В': 'В',
    'Г': 'Г',
    'Д': 'Д',
    'Е': 'Е',
    'Ж': 'Ж',
    'З': 'З',
    'И': 'И',
    'Й': 'Й',
    'К': 'К',
    'Л': 'Л',
    'М': 'М',
    'Н': 'Н',
    'О': 'О',
    'П': 'П',
    'Р': 'Р',
    'С': 'С',
    'Т': 'Т',
    'У': 'У',
    'Ф': 'Ф',
    'Х': 'Х',
    'Ц': 'Ц',
    'Ч': 'Ч',
    'Ш': 'Ш',
    'Щ': 'Щ',
    'Ъ': 'Ъ',
    'Ы': 'Ы',
    'Ь': 'Ь',
    'Э': 'Э',
    'Ю': 'Ю',
    'Я': 'Я',
    'A': 'А',
    'B': 'В',
    'C': 'С',
    'D': 'D',
    'E': 'Е',
    'F': 'F',
    'G': 'G',
    'H': 'Н',
    'I': 'I',
    'J': 'J',
    'K': 'К',
    'L': 'L',
    'M': 'М',
    'N': 'N',
    'O': 'О',
    'P': 'Р',
    'Q': 'Q',
    'R': 'R',
    'S': 'S',
    'T': 'Т',
    'U': 'U',
    'V': 'V',
    'W': 'W',
    'X': 'Х',
    'Y': 'У',
    'Z': 'Z',
    '0': 'О',
    '1': '1',
    '2': '2',
    '3': 'З',
    '4': '4',
    '5': '5',
    '6': 'б',
    '7': '7',
    '8': '8',
    '9': '9',
    '.': 'dot',
    ',': 'dot',
    '%': '%',
    '/': 'slash',
    '-': '-',
    '№': 'number',
    ':': 'colon',
    '+': '+',
    '*': '*',
    'а': 'а',
    'б': 'б',
    'в': 'В',
    'г': 'Г',
    'д': 'Д',
    'е': 'е',
    'ж': 'Ж',
    'з': 'З',
    'и': 'И',
    'й': 'Й',
    'к': 'К',
    'л': 'Л',
    'м': 'М',
    'н': 'Н',
    'о': 'О',
    'п': 'П',
    'р': 'Р',
    'с': 'С',
    'т': 'Т',
    'у': 'У',
    'ф': 'Ф',
    'х': 'Х',
    'ц': 'Ц',
    'ч': 'Ч',
    'ш': 'Ш',
    'щ': 'Щ',
    'ъ': 'Ъ',
    'ы': 'Ы',
    'ь': 'Ь',
    'э': 'Э',
    'ю': 'Ю',
    'я': 'Я',
    'a': 'а',
    'b': 'b',
    'c': 'С',
    'd': 'd',
    'e': 'e',
    'f': 'f',
    'g': 'g',
    'h': 'h',
    'i': 'i',
    'j': 'j',
    'k': 'К',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'o': 'О',
    'p': 'Р',
    'q': 'q',
    'r': 'r',
    's': 'S',
    't': 't',
    'u': 'U',
    'v': 'V',
    'w': 'W',
    'x': 'Х',
    'y': 'У',
    'z': 'Z', 
    '@': '@',
    ' ': 'space'
}

Point = namedtuple('Point', ['x', 'y']) 
Size  = namedtuple('Size', ['width', 'height'])

def parse_args():
      def existed_file(value):
            if not path.exists(value):
                  raise Exception('File not found: {!r}'.format(value))
            return value

      parser = ArgumentParser(description = "Tool for croping symbols boxes "\
                                            "from dataset")

      parser.add_argument('data_path', type=existed_file,
                          help='Path to dataset')
      parser.add_argument('files_dir', type=existed_file, 
                          help='Path to directory with .box files')
      out_def = path.join(path.abspath(path.dirname(__file__)), "marking")
      parser.add_argument('--outdir', type=str, default = out_def,
                          help='out directory (default - %s)'%(out_def))
      parser.add_argument('--good_bad', action="store_true", 
                          help='store symbols as good/bad instead of splitiing'
                          'by value')
      parser.add_argument('--h_mark', type=int, default=-1,
                          help='Height of the image, whicj have been used in'
                          'boxes extraction. -1 if disabled (default - '
                          'disabled)')
      
      return parser.parse_args()

    
def get_marking(data_path, files_dir, outdir, good_bad=False, h_mark=-1):
      list_dirs = []

      print(outdir)
      if path.exists(outdir):
            shutil.rmtree(outdir)
      makedirs(outdir)
      
      for value in DICT.values():
        if not list_dirs.count(value):
              list_dirs.append(value)
              
      marks = listdir(files_dir)
      
      ind = 0
      for mark in marks:
            with open(path.join(files_dir, mark), 'r') as in_file:
                  lines     = in_file.readlines()
            img = cv2.imread(path.join(data_path, 
                                       path.splitext(mark)[0] + '.jpg'))
            
            if h_mark != -1:
                img, scale = resize_h(img, 500)
            
            for i, line in enumerate(lines):
                  symb_name, bl_x, bl_y = None, None, None
                  tr_x, tr_y, null = None, None, None
                  if (len(line) and line[0] == ' '):
                      bl_x, bl_y, tr_x, tr_y, null = line.split()
                  else:
                      symb_name, bl_x, bl_y, tr_x, tr_y, null = line.split()
                      
                  bl_x = int(bl_x)
                  bl_y = int(img.shape[0] - int(bl_y))
                  tr_x = int(tr_x)
                  tr_y = int(img.shape[0] - int(tr_y))
                  
                  try:
                        #print (mark, symb_name)
                        symb = DICT[symb_name]
                        if good_bad:
                            if (symb == " " or symb == "@"):
                                symb = "bad"
                            else:
                                symb = "good"
                  except KeyError as e:
                        symb = 'bad'
                        
                  out_symb_dir = path.join(outdir, symb)
                  if not path.isdir(out_symb_dir):
                        makedirs(out_symb_dir)
                        
                  box = img[tr_y:bl_y, bl_x:tr_x]
                  cv2.imwrite(path.join(out_symb_dir, "%d.jpg"%(ind)), box)
                  ind += 1

      return list_dirs

      
if __name__ == '__main__':
      args = parse_args()
      get_marking(args.data_path, args.files_dir, args.outdir,
                  good_bad=args.good_bad, h_mark=args.h_mark)


















