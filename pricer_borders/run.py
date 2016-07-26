import copy
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np

from windows import CannyWindow, HoughWindow
from utils.os_utils import ask_image_path
from utils.filters import IMAGE_SIZE

def parse_args():
    def file_arg(value): 
        if not os.path.exists(value):
            if not os.path.exists(value):
                raise Exception() 
        return value
        
    if len(sys.argv) < 2:
        path = ask_image_path()
        sys.argv.append(path)
    parser = ArgumentParser()
    parser.add_argument('img', type=file_arg)
    return parser.parse_args()


if __name__ == '__main__': 
    args = parse_args()
    img = cv2.imread(args.img)
    img = cv2.resize(img, IMAGE_SIZE, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    canny_window = CannyWindow()
    hough_window = HoughWindow()
    while True:
        canny_params = canny_window.get()
        canny = cv2.Canny(gray, canny_params.thresh_1, 
                          canny_params.thresh_2)
        
        hough_params = hough_window.get()
        lines = cv2.HoughLinesP(canny, 1, np.pi/180, 
                                hough_params.thresh, 
                                hough_params.min_len, 
                                hough_params.max_gap)
        draw_frame = copy.copy(img)
        if lines is not None: 
            for x1, y1, x2, y2 in lines[0]:
                cv2.line(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        canny_window.show(canny)
        hough_window.show(draw_frame)
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        
    cv2.destroyAllWindows()
    
