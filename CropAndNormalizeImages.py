import cv2
import numpy as np
import shutil
from os import path, makedirs, environ

def CropAndNormalize(data_path, local_file_name, out_path, resolution):
	if(path.exists(out_path)):
		ans = raw_input('Path already exist. Would you like delete it? y/n \n')
		if ans == 'y':
			shutil.rmtree(out_path)
		else:
			print 'Specify a different path'
			return
	makedirs(out_path)

	resolution_w = resolution[0]
	resolution_h = resolution[1]

	local_file_path = '/'.join([data_path, local_file_name])
	local_file = open(local_file_path, 'r')
	inp = local_file.readline().split()

	while(len(inp) != 0):
		img_name = inp[0]
		point1, point2, point3, point4 = inp[1:3], inp[3:5], inp[5:7], inp[7:9]

		img_path = '/'.join([data_path, img_name])
		print img_path
		img = cv2.imread(img_path)
		img_h, img_w, ch  = img.shape

		pts1 = np.float32([point1, point2, point4, point3])
		pts2 = np.float32([[0, 0], [resolution_w, 0], [0, resolution_h], [resolution_w, resolution_h]])
		matrix = cv2.getPerspectiveTransform(pts1, pts2)
		img = cv2.warpPerspective(img, matrix,(img_h, img_w))

		img = img[0:resolution_h + 1, 0:resolution_w + 1]

		out_img_path = '/'.join([	out_path, img_name])
		cv2.imwrite(out_img_path ,img)

		inp = local_file.readline().split()

	local_file.close()

if __name__ == '__main__':
	DATA_PATH = environ['BEORGDATA'] + '/MarkedPricers'
	LOCAL_FILE_NAME = 'local_data.txt'
	OUT_PATH = environ['BEORGDATAGEN'] + '/CropedImages/'
	CropAndNormalize(DATA_PATH, LOCAL_FILE_NAME, OUT_PATH, (500, 250) )


