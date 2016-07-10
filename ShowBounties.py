from os import path, environ, makedirs
import shutil
import cv2
import numpy as np

def process_geom_file(data_path, file_name, out_path):
	file_path = '/'.join([data_path, file_name])
	folder = path.dirname(file_path)

	if(path.exists(out_path)):
		ans = raw_input('Path already exist. Would you like delete it? y/n \n')
		if ans == 'y':
			shutil.rmtree(out_path)
		else:
			print 'Specify a different path'
			return
	makedirs(out_path)
	lfile = open(file_path, 'r')
	for line in lfile.readlines():
		img_name, tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = line.split()
		img_path = '/'.join([data_path, img_name])
		img = cv2.imread(img_path)
		contour = np.array([ (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)], np.int)
		cv2.drawContours(img, [contour], -1, (0,255,0), cv2.CHAIN_APPROX_TC89_KCOS)
		img_outpath = '/'.join([out_path, img_name])
		cv2.imwrite(img_outpath, img)
	lfile.close()

if __name__ == '__main__':
	DATA_PATH = environ['BEORGDATAGEN'] + '/FinalData'
	OUT_PATH = environ['BEORGDATAGEN'] + '/RubliContour'
	process_geom_file(DATA_PATH, "rubli.txt", OUT_PATH)
	OUT_PATH = environ['BEORGDATAGEN'] + '/KopeikiContour'
	process_geom_file(DATA_PATH, "kopeiki.txt", OUT_PATH)
	OUT_PATH = environ['BEORGDATAGEN'] + '/NameContour'
	process_geom_file(DATA_PATH, "name.txt", OUT_PATH)
