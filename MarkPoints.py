import cv2
import numpy as np
import shutil
import os
import sys

#BE CAREFULL: GLOBAL LIST!
_PricerImageRect = []

def rectCallback(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONUP:
		_PricerImageRect.append((x, y))

def askPoints(numPoints, data_path, image_name, output_file):
	global _PricerImageRect
	image_path = os.path.join(data_path, image_name)
	_PricerImageRect = []
	window_name = 'Mark ' + str(numPoints) + ' points'
	cv2.namedWindow(window_name)
	cv2.setMouseCallback(window_name, rectCallback)
	while(len(_PricerImageRect) < numPoints):
		image = cv2.imread(image_path)
		for cords in _PricerImageRect:
			cv2.circle(image, cords, 5, (0, 255, 0), -1)
		cv2.imshow(window_name, image)
		k = cv2.waitKey(100) & 0xFF
		if k == 27: 		#ESC
			return -1
		if k == ord('c'):		#Cancel
			if(len(_PricerImageRect) != 0):
				del _PricerImageRect[len(_PricerImageRect) - 1]
		if k == ord('s'):  #Print coords
			print _PricerImageRect

	out_string = ' '.join([image_name, str(_PricerImageRect[0][0]), str(_PricerImageRect[0][1]), str(_PricerImageRect[1][0]), 
	str(_PricerImageRect[1][1]), str(_PricerImageRect[2][0]), str(_PricerImageRect[2][1]), str(_PricerImageRect[3][0]), str(_PricerImageRect[3][1]), '\n'])
	with open(output_file, 'a') as outfile:
		outfile.write(out_string)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	OUT_FILE_NAME = os.environ['BEORGDATA'] + '/ToMark/rubli.txt'
	DATA_PATH = os.environ['BEORGDATA'] + '/ToMark'

	out_file = open(os.path.join(DATA_PATH, OUT_FILE_NAME), 'w')
	out_file.close()

	files = os.listdir(DATA_PATH)
	img_list = filter(lambda x: x.endswith('.jpg'), files)

	for img in img_list:
		if (askPoints(4, DATA_PATH, img, OUT_FILE_NAME) == -1):
			sys.exit()
