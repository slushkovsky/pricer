import shutil
import cv2
import numpy as np
import h5py
import os
import re
from collections import namedtuple
from argparse import ArgumentParser

LINE_RE = re.compile('^ *(?P<img_name>\\S+) (?P<x>\\S+) *(?P<y>\\S+) *$')

Size  = namedtuple('Size', ['width', 'height'])

def parse_args():

      def size(value):
        m = re.match('(?P<width>\d+)x(?P<height>\d+)', value.strip())
        if not m:
            raise ValueError('Size not supported: {!r} (format: WxH)')            
        return (int(m.group('width')), int(m.group('height')))

      def existed_file(value):
            if not os.path.exists(value):
                  raise ValueError('File not found: {!r}'.format(value))
            return value

      parser = ArgumentParser(description = 'Tool for convertation dataset to hdf5')

      parser.add_argument('data_path',       type=existed_file,                      help='Path to dataset')
      parser.add_argument('file_path',       type=existed_file,                      help='Path to file with name of file and retailer')
      parser.add_argument('resize_to',       type=size,          default=(30,15),    help='Size to which we want to resize')
      parser.add_argument('n_chan',          type=int,           default=3,          help='Number of channels')
      parser.add_argument('test_period',     type=int,           default=10,         help='The percent of all data that should be tested')
      parser.add_argument('normalize_flag',  type=bool,          default=True,       help='Set up this flag to true if you want to normalize the source data') 

      return parser.parse_args()


def generate_HDF5(data_path, file_path, resize_to, n_chan,
                  test_period, normalize_flag,
                  out_dir=os.path.dirname(os.path.realpath(__file__))):
    out_name_train = os.path.join(out_dir, 'train.hdf5')
    out_name_test = os.path.join(out_dir, 'test.hdf5')
    indexer_train = os.path.join(out_dir, 'train.index.txt')
    indexer_test = os.path.join(out_dir, 'test.index.txt')
    
    
    with open(file_path, 'r') as in_file:
          lines = in_file.readlines()
          num_lines = len(lines)

    test_size   = int(num_lines / test_period)
    train_size  = num_lines - test_size
    data_train  = np.zeros((train_size, n_chan, resize_to[1], resize_to[0]),
                           dtype='f4')
    label_train = np.zeros((train_size, 2), dtype='f4')
    data_test   = np.zeros((test_size, n_chan, resize_to[1], resize_to[0]),
                           dtype='f4')
    label_test  = np.zeros((test_size, 2), dtype='f4')

    perc_done = 0
    for i, line in enumerate(lines):
          if i > num_lines*perc_done:
              perc_done += 0.01
              print(("%s percent done")%(int(perc_done * 100)))
       
         
          m = LINE_RE.match(line)
          if m is None: 
                raise Exception('Wrong line {}'.format(i))
          img_name = m.group('img_name')
  
          img_path = os.path.join(data_path, img_name)
          #print (img_path)
          img = cv2.imread(img_path)
           
          label = [float(m.group('x'))/img.shape[1],
                   float(m.group('y'))/img.shape[0]]
           
          if n_chan == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          size = Size(width=img.shape[1], height=img.shape[0])
          img = cv2.resize(img, resize_to, interpolation=cv2.INTER_CUBIC)
          if normalize_flag == True:
                img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
          if n_chan == 1:
                temp_img = img
                img = np.empty(temp_img.shape + (1,), dtype=temp_img.dtype)
                for y in range(temp_img.shape[0]):
                      for x in range(temp_img.shape[1]):
                            img[y][x] = [temp_img[y][x]]
          img = img.transpose((2, 0, 1))
   
          if (i + 1) % test_period == 0:
                data_test[i // test_period] = img
                label_test[i // test_period] = label
          else:
                data_train[i - i // test_period] = img
                label_train[i - i // test_period] = label

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
             
    with h5py.File(out_name_train, 'w') as out_train:
          out_train.create_dataset('data', data=data_train)
          out_train.create_dataset('label', data=label_train)

    with h5py.File(out_name_test, 'w') as out_test:
          out_test.create_dataset('data', data=data_test)
          out_test.create_dataset('label', data=label_test)

    with open(indexer_train, 'w') as lst_file_train:
          lst_file_train.write(out_name_train)

    with open(indexer_test, 'w') as lst_file_test:
          lst_file_test.write(out_name_test)

           
if __name__ == '__main__':
     args = parse_args()
     generate_HDF5(args.data_path, args.file_path, args.resize_to, args.n_chan, args.test_period, args.normalize_flag)


















