import cv2
import numpy as np
from math import atan2,degrees

def find_min_rect(src):
    """ Нахождение максимального вписанного прямоугольника 
    
    код переписан на python из
    https://stackoverflow.com/questions/32674256/how-to-adapt-or-resize-a-rectangle-inside-an-object-without-including-or-with-a
    
    @param $src - Матрица маски в формате OpenCV.
    
    Возвращает прямоугольник в формате (x1, y1, width, lenght)
    
    """
    #снижение разрешения маски для ускорения поиска
    reduction = 4
    src = cv2.resize(src,
                     ( int(src.shape[1]/reduction),
                       int(src.shape[0]/reduction)),
                     interpolation=cv2.INTER_NEAREST)   
    src = cv2.bitwise_not(src)
    w_mat = np.zeros(src.shape, dtype=np.uint)
    h_mat = np.zeros(src.shape, dtype=np.uint)
    max_rect = (0,0,0,0)
    max_area = float(0)
    for r in range(0, src.shape[0]):
        for c in range(0, src.shape[1]):
            if src[r,c] == 0:
                h_mat[r, c] = 1. + (h_mat[r-1, c] if (r>0) else 0)
                w_mat[r, c] = 1. + (w_mat[r, c-1] if (c>0) else 0)
            min_w = w_mat[r,c]
            for h in range(0, h_mat[r, c]):
                min_w = min(min_w, w_mat[r-h, c])
                area = (h+1) * min_w
                if area > max_area:
                    max_area = area
                    max_rect = (c - min_w + 1,
                               r - h, 
                               (c+1) - (c - min_w + 1),
                               (r+1) - (r - h))                
    return tuple(int(reduction*x) for x in max_rect)
    

def angle_between_points(x1, y1, x2, y2):
        x_diff = x2 - x1
        y_diff = y2 - y1
        return degrees(atan2(x_diff, y_diff))

    

