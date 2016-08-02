#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 20:34:23 2016

@author: chernov
"""

import random
import os
from argparse import ArgumentParser

import cv2
import numpy as np

random.seed()

__DEBUG__ = False

def increase_width_mirror(img, w_step, h_step):
    img_new = np.zeros((img.shape[0] + 2 * h_step,
                        img.shape[1] + 2 * w_step, 
                        img.shape[2]), np.uint8)
    
    img_new[h_step: h_step + img.shape[0],
            w_step: w_step + img.shape[1], :] = img

    w_new = img.shape[1] + w_step
    h_new = img.shape[0] + h_step

    img_new[:, 0:w_step, :] = img_new[:, w_step:2*w_step, :][:, ::-1, :]
    img_new[:, w_new:w_new + w_step,
            :] = img_new[:, w_new - w_step:w_new, :][:, ::-1, :]
    img_new[0:h_step, :, :] = img_new[h_step:2*h_step, :, :][::-1, :, :]
    img_new[h_new:h_new + h_step, :,
            :] = img_new[h_new - h_step:h_new, :, :][::-1, :, :]
    return img_new
    
    
def nomalize_rect(rect):
    if rect.shape[0] != 4 or rect.shape[1] != 2:
        raise Exception()
    
    center = tuple((rect.sum(axis=0)/rect.shape[0]).astype(np.int))
    
    rect_normalized = np.zeros(rect.shape, np.int)
    for corner in rect:
        if corner[0] < center[0] and corner[1] < center[1]:
            rect_normalized[0] = corner
        elif corner[0] > center[0] and corner[1] < center[1]:
            rect_normalized[1] = corner
        elif corner[0] > center[0] and corner[1] > center[1]:
            rect_normalized[2] = corner
        elif corner[0] < center[0] and corner[1] > center[1]:
            rect_normalized[3] = corner

    if((rect_normalized.sum(axis=1) == 0).any()):
        return None
    return rect_normalized
    
def crop_pricer(img, rect, h_crop_perc=0.3, max_rotation=10):
    rect = nomalize_rect(rect)
    
    if type(rect) == type(None):
        return None, None
    
    min_x = rect[(0,3),0].max()
    max_x = rect[(1,2),0].min()
    min_y = rect[(0,1),1].max()
    max_y = rect[(2,3),1].min()

    h_crop_pixels = int(img.shape[0]*h_crop_perc)
    offset = random.randint(
    
    if __DEBUG__:
        draw_img = img.copy()
        rect_ins = np.array(([min_x, min_y], [max_x, min_y],
                             [max_x, max_y], [min_x, max_y]))
        cv2.drawContours(draw_img, [rect], -1, (255, 0, 0))
        cv2.drawContours(draw_img, [rect_ins], -1, (0, 255, 0))
        cv2.imshow("rects", draw_img)
        
    x1 = random.randint(min_x, 
                        min_x + int((max_x - min_x) * (1.0 - reserve_perc)))
    y1 = random.randint(-int(img.shape[1]*h_crop_perc),
                        int(img.shape[1]*h_crop_perc))
    x_offset_max = int((img.shape[1] - rect[:,0].max()) * (1.0 - reserve_perc))
    x2 = img.shape[1] + random.randint(-x_offset_max, x_offset_max)
    y2 = random.randint(min_y + int((max_y - min_y)*reserve_perc),
                        max_y)
    
    crop_rect = np.array(([x1, y1], [x2, y1], [x2, y2], [x1, y2]))
    if __DEBUG__:
        draw_img = img.copy()
        cv2.drawContours(draw_img, [crop_rect], -1, (0, 255, 0))
        cv2.imshow("crop_rect", draw_img)
    
    crop_rect_rot = None
    k = 0
    while True:
        rotation = random.uniform(-max_rotation/2, max_rotation/2)
        rot  = cv2.getRotationMatrix2D((x2, y2),rotation, 1.0)    
        crop_rect_rot = np.dot(crop_rect, rot)[:, 0:2].astype(np.int)
        if cv2.pointPolygonTest(crop_rect_rot, tuple(rect[1]), False) > 0:
            break
        else:
            k += 1
            if k > rot_max_tries:
                return None, None
    
    min_x = crop_rect_rot[:,0].min()
    max_x = crop_rect_rot[:,0].max()
    min_y = crop_rect_rot[:,1].min()
    max_y = crop_rect_rot[:,1].max()
    
    overlap_x = 0
    if min_x < 0:
        overlap_x = abs(min_x)
    if max_x > img.shape[1]:
        overlap_x = max(overlap_x, max_x - img.shape[1])
    overlap_y = 0
    if min_y < 0:
        overlap_y = abs(min_y)
    if max_y > img.shape[0]:
        overlap_y = max(overlap_y, max_y - img.shape[0])
    
    img = increase_width_mirror(img, overlap_x, overlap_y)
    crop_rect_rot = crop_rect_rot + (overlap_x, overlap_y)
    if __DEBUG__:
        draw_img = img.copy()
        cv2.drawContours(draw_img, [crop_rect_rot], -1, (0, 255, 0))
        cv2.imshow("crop_rect_rot", draw_img)
    
    warped_w = max_x - min_x
    warped_h = max_y - min_y
    
    warped_rect = np.array(([0, 0], [warped_w, 0],
                            [warped_w, warped_h], [0,warped_h]))
    
    M = cv2.getPerspectiveTransform(crop_rect_rot.astype(np.float32),
                                    warped_rect.astype(np.float32))
    warped = cv2.warpPerspective(img, M, (warped_w, warped_h))
    
    corner = np.array([[rect[1] + (overlap_x, overlap_y)]], np.float32)
    corner_warped = cv2.perspectiveTransform(corner, M).ravel()
    if __DEBUG__:
        draw_img = warped.copy()
        cv2.circle(draw_img, tuple(corner_warped), 3, (0,0,255), 2)
        cv2.imshow("warped", draw_img)
        
    return warped, corner_warped
    

def crop_quadrant(img, rect, h_crop_perc=0.3,
                    reserve_perc=0.3, max_rotation=10, rot_max_tries=100):
    
    rect = nomalize_rect(rect)
    
    if type(rect) == type(None):
        return None, None
    
    min_x = rect[(0,3),0].max()
    max_x = rect[(1,2),0].min()
    min_y = rect[(0,1),1].max()
    max_y = rect[(2,3),1].min()
    
    if __DEBUG__:
        draw_img = img.copy()
        rect_ins = np.array(([min_x, min_y], [max_x, min_y],
                             [max_x, max_y], [min_x, max_y]))
        cv2.drawContours(draw_img, [rect], -1, (255, 0, 0))
        cv2.drawContours(draw_img, [rect_ins], -1, (0, 255, 0))
        cv2.imshow("rects", draw_img)
        
    x1 = random.randint(min_x, 
                        min_x + int((max_x - min_x) * (1.0 - reserve_perc)))
    y1 = random.randint(-int(img.shape[1]*h_crop_perc),
                        int(img.shape[1]*h_crop_perc))
    x_offset_max = int((img.shape[1] - rect[:,0].max()) * (1.0 - reserve_perc))
    x2 = img.shape[1] + random.randint(-x_offset_max, x_offset_max)
    y2 = random.randint(min_y + int((max_y - min_y)*reserve_perc),
                        max_y)
    
    crop_rect = np.array(([x1, y1], [x2, y1], [x2, y2], [x1, y2]))
    if __DEBUG__:
        draw_img = img.copy()
        cv2.drawContours(draw_img, [crop_rect], -1, (0, 255, 0))
        cv2.imshow("crop_rect", draw_img)
    
    crop_rect_rot = None
    k = 0
    while True:
        rotation = random.uniform(-max_rotation/2, max_rotation/2)
        rot  = cv2.getRotationMatrix2D((x2, y2),rotation, 1.0)    
        crop_rect_rot = np.dot(crop_rect, rot)[:, 0:2].astype(np.int)
        if cv2.pointPolygonTest(crop_rect_rot, tuple(rect[1]), False) > 0:
            break
        else:
            k += 1
            if k > rot_max_tries:
                return None, None
    
    min_x = crop_rect_rot[:,0].min()
    max_x = crop_rect_rot[:,0].max()
    min_y = crop_rect_rot[:,1].min()
    max_y = crop_rect_rot[:,1].max()
    
    overlap_x = 0
    if min_x < 0:
        overlap_x = abs(min_x)
    if max_x > img.shape[1]:
        overlap_x = max(overlap_x, max_x - img.shape[1])
    overlap_y = 0
    if min_y < 0:
        overlap_y = abs(min_y)
    if max_y > img.shape[0]:
        overlap_y = max(overlap_y, max_y - img.shape[0])
    
    img = increase_width_mirror(img, overlap_x, overlap_y)
    crop_rect_rot = crop_rect_rot + (overlap_x, overlap_y)
    if __DEBUG__:
        draw_img = img.copy()
        cv2.drawContours(draw_img, [crop_rect_rot], -1, (0, 255, 0))
        cv2.imshow("crop_rect_rot", draw_img)
    
    warped_w = max_x - min_x
    warped_h = max_y - min_y
    
    warped_rect = np.array(([0, 0], [warped_w, 0],
                            [warped_w, warped_h], [0,warped_h]))
    
    M = cv2.getPerspectiveTransform(crop_rect_rot.astype(np.float32),
                                    warped_rect.astype(np.float32))
    warped = cv2.warpPerspective(img, M, (warped_w, warped_h))
    
    corner = np.array([[rect[1] + (overlap_x, overlap_y)]], np.float32)
    corner_warped = cv2.perspectiveTransform(corner, M).ravel()
    if __DEBUG__:
        draw_img = warped.copy()
        cv2.circle(draw_img, tuple(corner_warped), 3, (0,0,255), 2)
        cv2.imshow("warped", draw_img)
        
    return warped, corner_warped
   

def parse_args():
    def file_arg(value): 
        if not os.path.exists(value):
            if not os.path.exists(value):
                raise Exception() 
        return value
        
    parser = ArgumentParser()
    parser.add_argument('image', type=file_arg,
                        help="Image filepath")
    parser.add_argument('p1_x', type=int,
                        help="coord")
    parser.add_argument('p1_y', type=int,
                        help="coord")
    parser.add_argument('p2_x', type=int,
                        help="coord")
    parser.add_argument('p2_y', type=int,
                        help="coord")
    parser.add_argument('p3_x', type=int,
                        help="coord")
    parser.add_argument('p3_y', type=int,
                        help="coord")
    parser.add_argument('p4_x', type=int,
                        help="coord")
    parser.add_argument('p4_y', type=int,
                        help="coord")
    parser.add_argument('--debug', action="store_true",
                        help="Debug mode")
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        __DEBUG__ = True
    
    img_path = args.image
    
    rect = np.array(([args.p1_x, args.p1_y], [args.p2_x, args.p2_y],
                     [args.p3_x, args.p3_y], [args.p4_x, args.p4_y]))
    
    img = cv2.imread(img_path)
    warped, point = crop_quadrant(img, rect)
    
    cv2.circle(warped, tuple(point), 3, (0,0,255), 2)
    cv2.imshow("warped", warped)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    