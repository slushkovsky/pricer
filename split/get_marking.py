import cv2
import argparse
import os
import sys
from math import sqrt
import numpy as np
from collections import namedtuple
import shutil

os.environ['GLOG_minloglevel'] = '2' 
# Caffe quiet (log levels: 0-debug, 1-all info, 2-warnings, 3-errors)
import caffe

from split_hist_cv import split_lines_hist, crop_regions
from split_symbols_cv import detect_text_cv
from split_ruble_symbols import process_image

INPUT_IMG_W = 30
INPUT_IMG_H = 60
INPUT_LAYER_NAME = 'data'
OUTPUT_LAYER_NAME = 'ip2'

FILES_END_WITH = '.jpg'

DEFAULT_OUT_NAME = './marking_box'
DEFAULT_RUB_MODEL = '../workspace/Pricer/Rubli60x30/Rubli60x30_iter_10000.caffemodel'
DEFAULT_RUB_CONF = '../workspace/Pricer/Rubli60x30/netconfig.prototxt'
DEFAULT_KOP_MODEL = '../workspace/Pricer/Kopeiki60x30/Kopeiki60x30_iter_10000.caffemodel'
DEFAULT_KOP_CONF = '../workspace/Pricer/Kopeiki60x30/netconfig.prototxt'
DEFAULT_NAME_MODEL = '../workspace/Pricer/Name60x30/Name60x30_iter_10000.caffemodel'
DEFAULT_NAME_CONF = '../workspace/Pricer/Name60x30/netconfig.prototxt'
DEFAULT_NM1 = 'pretrained_classifiers/trained_classifierNM1.xml'
DEFAULT_NM2 = 'pretrained_classifiers/trained_classifierNM2.xml'

def progress(progress, pref = '', suf = '', new_line = 0):
	progress = int(100 * progress)
	bar = '|' * progress + '.' * (100 - progress)
	sys.stdout.write('%s%s%d%%%s\r' % (pref, bar, progress, suf))
	if progress == 100 or new_line == 1:
		sys.stdout.write("%s%s%d%%%s\n" % (pref, bar, progress, suf))

def ask_localization_net(img, model_path, config_path):
	Model = namedtuple('Model', ['struct', 'weights'])
	model = Model(config_path, model_path)
	net = caffe.Net(model.struct, model.weights, caffe.TEST)
	
	transformer = caffe.io.Transformer(	
			{INPUT_LAYER_NAME: net.blobs[INPUT_LAYER_NAME].data.shape})
	transformer.set_transpose(INPUT_LAYER_NAME, (2, 0, 1))
	transformer.set_raw_scale(INPUT_LAYER_NAME, 255)

	transformed_img = transformer.preprocess(INPUT_LAYER_NAME, img)
	net.blobs[INPUT_LAYER_NAME].data[...] = transformed_img
	res = net.forward()[OUTPUT_LAYER_NAME][0]
	return res

def resize_img_h(img, new_h=250):
	scale = new_h / img.shape[0]
	new_size = np.array(img.shape) * scale 
	new_size = tuple(np.flipud(new_size[0:2]).astype(np.int))
	new_img = cv2.resize(img, new_size)
	return new_img, scale

def extract_symbol_rects(img, args = None, offset=(0,0)):
	nm1, nm2 = args
	rects, rects_bad = [], []
	regions = crop_regions(split_lines_hist(img), img.shape[0] * 0.15)   
	for region in regions:
		cur_img_part = img[region[0]:region[1] , :]
		cur_rects, cur_rects_bad = detect_text_cv(cur_img_part,
															nm1, nm2)
		for rect in cur_rects:
			rect[1] += region[0] + offset[1]
			rect[0] += offset[0]
			rects.append(rect)
		for rect in cur_rects_bad:
			rect[1] += region[0] + offset[1]
			rect[0] += offset[0]
			rects_bad.append(rect)
	return rects, rects_bad

def extract_rubli_rects(img, args = None, offset=(0,0)):
	rects = process_image(img)
	for i in range(len(rects)):
		rects[i][0] += offset[0]
		rects[i][1] += offset[1]
	return rects


def extract_debug(img, args, offset):
	return [(0, 0, 0, 0)]

def clear_file(path):
	f = open(path, 'w')
	f.close()

def save_marking_diff_files(out_path, coordinates, name = '', suf = ''
													, need_clear = 0):
	if need_clear:
		if(os.path.exists(out_path)):
			ans = raw_input('Path already exist.'+ 
									'Would you like delete it? y/n \n')
			if ans == 'y':
				shutil.rmtree(out_path)
			else:
				print 'Specify a different path'
				return
		os.makedirs(out_path)
		return
	name += '.box'
	path = os.path.join(out_path, name)
	x0, y0, x1, y1 = coordinates
	coords_str = ' '.join([str(x0), str(y0), str(x1), str(y1)])
	out_str = ' '.join(['*', coords_str, '0', suf]) + '\n'
	with open(path, 'a') as out_file:
		out_file.write(out_str)

def save_marking(path, coordinates, pref = '', suf = '' 
													, need_clear = 0):
	if need_clear:
		clear_file(path)
		return
	x0, y0, x1, y1 = coordinates
	coords_str = ' '.join([str(x0), str(y0), str(x1), str(y1)])
	out_str = ' '.join([pref, '*', coords_str, '0', suf]) + '\n'
	with open(path, 'a') as out_file:
		out_file.write(out_str)

def imgCrop(img, bounties):
	tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = bounties
	#Compute approximate width and height of the image to correct perspective 
	width_y = max((tr_y - tl_y)**2, (br_y - bl_y)**2)
	width_x = max((tr_x - tl_x)**2, (br_x - bl_x)**2)
	width = int(sqrt(width_y + width_x))
	height_y = max((tl_y - bl_y)**2, (tr_y - br_y)**2)
	height_x = max((bl_x - tl_x)**2, (br_x - tl_x)**2)
	height = int(sqrt(height_y + height_x))
	#Correcting perspective and cropping the image
	pts1 = np.float32([[tl_x, tl_y], [tr_x, tr_y], [bl_x, bl_y], [br_x, br_y]])
	pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
	matrix = cv2.getPerspectiveTransform(pts1, pts2)
	img = cv2.warpPerspective(img, matrix, (width, height))
	return img


def proccess_marking(img, model_path, config_path, out_path
					, extract_func, save_func 
									, args = None, pref = '', suf = ''):
	coordinates = ask_localization_net(img, model_path, config_path)
	x0, y0, x1, y1, x2, y2, x3, y4 = coordinates
	x0 *= img.shape[1]
	x1 *= img.shape[1]
	x2 *= img.shape[1]
	x3 *= img.shape[1]
	y0 *= img.shape[0]
	y1 *= img.shape[0]
	y2 *= img.shape[0]
	y3 *= img.shape[0]
	coordinates = x0, y0, x1, y1, x2, y2, x3, y4
	crop = imgCrop(img, coordinates)
	rects = extract_func(crop, args, offset = (x0, y0))
	for rect in rects:
		save_func(out_path, rect, pref, suf)

def proccess_rkn(data_path, rubli_model_path, rubli_config_path
								,kopeiki_model_path, kopeiki_config_path
								,name_model_path, name_config_path
								,classifiers ,out_path
								, save_func , unpr = False):
	files = os.listdir(data_path)
	inter = filter(lambda x: x.endswith(FILES_END_WITH), files)
	save_func(out_path, (0, 0, 0, 0)
										, need_clear = 1)	#clear file
	for i in range(len(inter)):
		if unpr:
			progress(float(i) / len(inter), pref = 'Loading  '
								, suf = ' (%d / %d)' % (i, len(inter)))
		img_path = os.path.join(data_path, inter[i])
		img = cv2.imread(img_path)
		proccess_marking(img, rubli_model_path, rubli_config_path
										, out_path, extract_rubli_rects
										, save_func
										, pref = inter[i], suf = 'r')
		#proccess_marking(img, kopeiki_model_path, kopeiki_config_path
			#, out_path, extract_rubli_rects, save_func
			#, pref = inter[i], suf = 'k')
		proccess_marking(img, name_model_path, name_config_path
					, out_path, extract_symbol_rects, save_func
					, args = classifiers, pref = inter[i], suf = 'n')

def mark_arguments():
	parser = argparse.ArgumentParser(description ='This script allows' +
			' you to run caffe localization net and save the result.')

	parser.add_argument('data_path', type = str,
			help = 'Enter path to images, which should be processed')

	parser.add_argument('--rub_model', type = str
										, default = DEFAULT_RUB_MODEL
					,help = 'Enter path to caffemodel of rubles net. ' +
									'Default is %s' % DEFAULT_RUB_MODEL)
	parser.add_argument('--rub_config', type = str
										, default = DEFAULT_RUB_CONF
				,help = 'Enter path to configure file of rubles net. ' +
									'Default is %s' % DEFAULT_RUB_CONF)
	parser.add_argument('--kop_model', type = str
										, default = DEFAULT_KOP_MODEL
				,help = 'Enter path to caffemodel of kopeiki net. ' +
									'Default is %s' % DEFAULT_KOP_MODEL)
	parser.add_argument('--kop_config', type = str
										, default = DEFAULT_KOP_CONF
			,help = 'Enter path to configure file of kopeiki net. ' +
									'Default is %s' % DEFAULT_KOP_CONF)
	parser.add_argument('--name_model', type = str
										, default = DEFAULT_NAME_MODEL
					,help = 'Enter path to caffemodel of name net. ' +
								'Default is %s' % DEFAULT_NAME_MODEL)
	parser.add_argument('--name_config', type = str
										, default = DEFAULT_NAME_CONF
			,help = 'Enter path to configure file of name net. ' +
									'Default is %s' % DEFAULT_NAME_CONF)

	parser.add_argument('--nm1', type = str
										, default = DEFAULT_NM1
			,help = 'Enter path to first classifier. ' +
									'Default is %s' % DEFAULT_NM1)
	parser.add_argument('--nm2', type = str
										, default = DEFAULT_NM2
			,help = 'Enter path to second classifier. ' +
									'Default is %s' % DEFAULT_NM2)

	parser.add_argument('--onef', action='store_true'
				, help = 'Use if you want to save output in one file. '+
							'This way, you should specify the out path')
	parser.add_argument('--unpr', action='store_false'
				, help = 'Use if you want to unable progress interface.')
	parser.add_argument('--out', type = str, default = DEFAULT_OUT_NAME,
				help = 'Enter path to output file. Default is %s'
													% DEFAULT_OUT_NAME)

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = mark_arguments()
	nm = (args.nm1, args.nm2)
	save_func = save_marking_diff_files
	if args.onef:
		save_func = save_marking
	try:
		proccess_rkn(args.data_path, args.rub_model, args.rub_config
									, args.kop_model, args.kop_config
									, args.name_model, args.name_config
											, nm, args.out
											, save_func, args.unpr)
	except KeyboardInterrupt:
		print()
