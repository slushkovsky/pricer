from os import path, makedirs
import shutil

import cv2
import numpy as np
import h5py


def draw_geometry(filepath):
    """Рисует контуры на изображениях.
    
    Изображения с контурами сохраняются в папке с выбокрой в подпапке batch
    
    Keyword arguments:
    filepath -- Путь к файлу с разметкой в формате
                https://advertle.slack.com/files/slushkov/F1LRWLR8X/local_data.txt
                Файл разметки должен находиться в директории с выборкой
    
    """
    folder = path.dirname(filepath)
    
    batch_dir = folder + "/batch"
    if(path.exists(batch_dir)):
        shutil.rmtree(batch_dir)
    makedirs(batch_dir)
    
    print(folder)
    with open(filepath) as file:
        for line in file.readlines():
            name, tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = line.split()
            image = cv2.imread(folder + "/" + name)
            contour = np.array([(tl_x, tl_y),
                                 (tr_x, tr_y),
                                 (br_x, br_y),
                                 (bl_x, bl_y)], np.int)
            cv2.drawContours(image, [contour], -1, (0,255,0),
                             cv2.CHAIN_APPROX_TC89_KCOS)
            cv2.imwrite(batch_dir + "/" + name, image)


def generate_h5_db(filepath, db_name, n_chan=3, image_size=(250,125), 
                   dataset_data_name='data', dataset_label_name='label'):
    """Создание hdf5 выборки для caffe из файла разметки.
    
    На входе выборки - пережатые под размер image_size изображения 
    с интенсивностью (0.0 - 1.0)
    На выходе - 8 float чисел, соответсвующие краям области
    
    Keyword arguments:
    filepath -- Путь к файлу с разметкой в формате
                https://advertle.slack.com/files/slushkov/F1LRWLR8X/local_data.txt
                Файл разметки должен находиться в директории с выборкой
    db_name -- Путь до создаваемой hdf5 бд.
    n_chan -- Количество каналов изображений.
    image_size -- Конечный размер изображений.
    dataset_data_name -- имя поля изображений в бд.
    dataset_label_name -- имя поля лейблов в бд.
                
    """
    folder = path.dirname(filepath)
    
    num_lines = sum(1 for line in open(filepath))
    
    data = np.zeros((num_lines, n_chan,
                     image_size[0], image_size[1]), dtype="f4")
    label = np.zeros((num_lines, 8), dtype='f4')
    
    with h5py.File(db_name, 'w') as db:
        with open(filepath) as file:
            i = 0
            for line in file.readlines():
                name, tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = line.split()
                
                img = cv2.imread(folder + "/" + name)
                size = img.shape
                img = cv2.resize(img, image_size, 
                                 interpolation = cv2.INTER_CUBIC)
                img = cv2.normalize(img.astype('float'), 
                                    None, 0.0, 1.0, cv2.NORM_MINMAX)
                
                img = img.transpose((2,1,0))
                
                contour = [int(tl_x) / size[1], 
                           int(tl_y) / size[0],
                           int(tr_x) / size[1], 
                           int(tr_y) / size[0],
                           int(br_x) / size[1], 
                           int(br_y) / size[0],
                           int(bl_x) / size[1],
                           int(bl_y) / size[0]]
            
                data[i] = img
                label[i] = contour
                i += 1

        db.create_dataset(dataset_data_name, data=data)
        db.create_dataset(dataset_label_name, data=label)
        db.close()
        
    with open( db_name + '.list.txt','w') as L:
        L.write(db_name)

