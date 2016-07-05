from os import path, makedirs
import shutil
import random
import copy

import cv2
import numpy as np
import h5py

from increase_data import increase_data


def draw_geometry(marking_path):
    """Рисует контуры на изображениях.
    
    Изображения с контурами сохраняются в папке с выбокрой в подпапке batch
    
    Keyword arguments:
    marking_path -- Путь к файлу с разметкой в формате
                https://advertle.slack.com/files/slushkov/F1LRWLR8X/local_data.txt
                Файл разметки должен находиться в директории с выборкой
    
    """
    folder = path.dirname(marking_path)
    
    batch_dir = folder + "/batch"
    if(path.exists(batch_dir)):
        shutil.rmtree(batch_dir)
    makedirs(batch_dir)
    
    print(folder)
    with open(marking_path) as file:
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
            
def _draw_contour(img, contour):
    """ Отрисовка контура на изображении
    
    Keyword arguments:
    img -- изображение
    contour -- контур с относительными координатами в формате
    [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]
    
    """
    tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = contour
    shape = img.shape
    
    contour_abs = np.array([(tl_x * shape[1], tl_y * shape[0]),
                            (tr_x * shape[1], tr_y * shape[0]),
                            (br_x * shape[1], br_y * shape[0]),
                            (bl_x * shape[1], bl_y * shape[0])], np.int)
    
    cv2.putText(img, "tl", tuple((contour_abs[0])), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (255,255,0))
    cv2.putText(img, "tr", tuple((contour_abs[1])), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (255,255,0))
    cv2.putText(img, "bl", tuple((contour_abs[2])), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (255,255,0))
    cv2.putText(img, "br", tuple((contour_abs[3])), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (255,255,0))
    
    cv2.drawContours(img, [contour_abs], -1, (0,255,0),
                     cv2.CHAIN_APPROX_TC89_KCOS)
    return img
    


def generate_h5_db(marking_path, images_path, 
                   db_name_train, db_name_test, test_percent=0.1,
                   increase_size=5,
                   n_chan=3,
                   image_size=(250,125),
                   original_image_size=(500, 250),
                   use_abs_coords = False,
                   dataset_data_name='data',
                   dataset_label_name='label',
                   test=False):
    """Создание hdf5 выборки для caffe из файла разметки.
    
    На входе выборки - пережатые под размер image_size изображения 
    с интенсивностью (0.0 - 1.0)
    На выходе - 8 float чисел, соответсвующие краям области
    
    Keyword arguments:
    marking_path -- Путь к файлу с разметкой в формате
                https://advertle.slack.com/files/slushkov/F1LRWLR8X/local_data.txt
    images_path -- путь до директории с изображениями
    db_name_train -- Путь до создаваемой тренировочной hdf5 бд.
    db_name_test -- Путь до создаваемой тестовой hdf5 бд.
    test_percent -- Процент изображений, которые пойдут в тестовую выборку.
    increase_size -- Количество искуственных изображений, создаваемых для 
    каждого оригинального
    n_chan -- Количество каналов изображений.
    image_size -- Конечный размер изображений.
    original_image_size -- размер исходных изображений, на которых проводилась
    разметка
    use_abs_coords -- использовать абсолютные координаты в лейбле
    dataset_data_name -- имя поля изображений в бд.
    dataset_label_name -- имя поля лейблов в бд.
    test -- тестирование функции
                
    """
    
    num_lines = sum(1 for line in open(marking_path))
    
    total_size = num_lines * (increase_size + 1)
    test_size = int(total_size * test_percent)
    test_samples = random.sample(range(total_size), test_size)
    
    
    data_train = np.zeros((total_size - test_size, n_chan,
                           image_size[0], image_size[1]), dtype="f4")
    label_train = np.zeros((total_size - test_size, 8), dtype='f4')
    data_test = np.zeros((test_size, n_chan,
                          image_size[0], image_size[1]), dtype="f4")
    label_test = np.zeros((test_size, 8), dtype='f4')
    
    
    with h5py.File(db_name_train, 'w') as db_train:
        with h5py.File(db_name_test, 'w') as db_test:
            with open(marking_path) as file:
                cur_total, cur_train, cur_test = 0, 0, 0
                for line in file.readlines():
                    values = line.split()
                    name, pts = values[0], [float(i) for i in values[1:]]

                    for i in range(0, len(pts), 2):
                        pts[i] /= original_image_size[0]
                        pts[i + 1] /= original_image_size[1]
                    
                    center = np.array((0,0), np.float)
                    for i in range(0, len(pts), 2):
                        center += np.array((pts[i], pts[i + 1]), np.float)
                    center /= len(pts)/2
                    
                    tl_x = tl_y = tr_x = tr_y = None
                    br_x = br_y = bl_x = bl_y = None
                    for i in range(0, len(pts), 2):
                        point = np.array((pts[i], pts[i + 1]), np.float)
                        diff = point - center
                        
                        if diff[0] < 0 and diff[1] < 0:
                            tl_x = point[0]
                            tl_y = point[1]
                        elif diff[0] > 0 and diff[1] < 0:
                            tr_x = point[0]
                            tr_y = point[1]
                        elif diff[0] > 0 and diff[1] > 0:
                            br_x = point[0]
                            br_y = point[1]
                        elif diff[0] < 0 and diff[1] > 0:
                            bl_x = point[0]
                            bl_y = point[1]
                    
                    contour = [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]
                    
                    img_orig = cv2.imread(images_path + "/" + name)
                    
                    increased_imgs = increase_data(img_orig, increase_size)
                    increased_imgs.append(img_orig)
                    for img in increased_imgs:
                        
                        if test:
                            draw_img = copy.copy(img)
                            draw_img = _draw_contour(img, contour)
                            draw_img = cv2.resize(draw_img, (500, 250))
                            cv2.imshow("contour", draw_img)
                            ret = cv2.waitKey() & 0xFF
                            if ret == 27:
                                cv2.destroyAllWindows()
                                test = False
                        
                        img = cv2.resize(img, image_size, 
                                         interpolation = cv2.INTER_CUBIC)
                        img = cv2.normalize(img.astype('float'), 
                                            None, 0.0, 1.0, cv2.NORM_MINMAX)
                        
                        img = img.transpose((2,1,0))
                        
                        contour_shaped = copy.copy(contour)
                        if use_abs_coords:
                            for i in range(0, len(contour_shaped), 2):
                                contour_shaped[i] *= img.shape[1]
                                contour_shaped[i + 1] *= img.shape[2]
                    
                        if cur_total in test_samples:
                            data_test[cur_test] = img
                            label_test[cur_test] = contour_shaped
                            cur_test += 1
                        else:
                            data_train[cur_train] = img
                            label_train[cur_train] = contour_shaped
                            cur_train += 1
                        cur_total += 1

            db_train.create_dataset(dataset_data_name, data=data_train)
            db_train.create_dataset(dataset_label_name, data=label_train)
            db_train.close()
            with open(db_name_train + '.list.txt','w') as L:
                L.write(db_name_train)
                
            db_test.create_dataset(dataset_data_name, data=data_test)
            db_test.create_dataset(dataset_label_name, data=label_test)
            db_test.close()
            with open(db_name_test + '.list.txt','w') as L:
                L.write(db_name_test)

