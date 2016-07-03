from os import path, makedirs
import shutil

import cv2
import numpy as np

def process_geom_file(filepath):
    folder = path.dirname(filepath)
    
    batch_dir = "batchK"
    if(path.exists(batch_dir)):
        shutil.rmtree(batch_dir)
    makedirs(batch_dir)
    
    print(folder)
    with open(filepath) as file:
        for line in file.readlines():
            name, tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = line.split()
            image = cv2.imread(folder + "/" + name)
            contour = np.array([ (tl_x, tl_y),
                                 (tr_x, tr_y),
                                 (br_x, br_y),
                                 (bl_x, bl_y)], np.int)
            cv2.drawContours(image, [contour], -1, (0,255,0),
                             cv2.CHAIN_APPROX_TC89_KCOS)
            cv2.imwrite(batch_dir + "/" + name, image)

process_geom_file("LData/kopeiki.txt")
