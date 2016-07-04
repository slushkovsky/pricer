import shutil
from os import path, makedirs, environ

import cv2
import numpy as np

MARK_DATA_PATH = environ["BEORGDATA"] + "/marked_pricers/local_data.txt"
DATA_PATH = environ["BEORGDATA"] + "/marked_pricers"
OUT_PATH = environ["BEORGDATAGEN"] + "/CData_full"

if(path.exists(OUT_PATH)):
    shutil.rmtree(OUT_PATH)
makedirs(OUT_PATH)

with open(MARK_DATA_PATH, 'r') as mark_file:
    for line in mark_file.readlines():
        s = line.split()
        print(s[0])
        img =  cv2.imread(DATA_PATH + "/" + s[0])
        rows,cols, ch  = img.shape
        
        p1, p2, p3, p4 = s[1:3], s[3:5], s[5:7], s[7:9]
        pts1 = np.float32([p1, p2, p4, p3])
        rect = cv2.minAreaRect(pts1)
        
        box = cv2.boxPoints(rect)
        start_from = 0 # начало контура не всегда в левом верхнем углу
        if abs(box[0][0] - box[1][0]) < abs(box[0][1] - box[1][1]):
            start_from = 1
            
        w = np.linalg.norm(np.array(box[start_from + 1]) -
                                    np.array(box[start_from + 0]))
        h = np.linalg.norm(np.array(box[start_from + 2]) -
                                    np.array(box[start_from + 1]))
        
        pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        img = cv2.warpPerspective(img,M,(cols, rows))
        img = img[0:h, 0:w]
        cv2.imwrite(OUT_PATH + '/' + s[0] ,img)
