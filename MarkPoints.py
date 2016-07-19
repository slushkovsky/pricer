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
	while(True):
		image = cv2.imread(image_path)
		size = len(_PricerImageRect)
		for i in range(size):
			cv2.circle(image, _PricerImageRect[i], 5, (0, 255, 0), -1)
			cv2.line(image, _PricerImageRect[i], _PricerImageRect[(i + 1) % size], (0, 255, 0))
		cv2.imshow(window_name, image)
		k = cv2.waitKey(100) & 0xFF
		if k == 27:				#ESC
			return -1
		if k == ord('p'):		#Print coords
			print _PricerImageRect
		if len(_PricerImageRect) >= numPoints:
			if k == 10:			#Enter
				break
		if(len(_PricerImageRect) != 0):
			if k == 8:	#Backspace
				del _PricerImageRect[len(_PricerImageRect) - 1]
			x, y = _PricerImageRect[len(_PricerImageRect) - 1]
			if k == 82: 		#UpKey
				_PricerImageRect[len(_PricerImageRect) - 1] = (x, y - 1)
			if k == 84: 		#DownKey
				_PricerImageRect[len(_PricerImageRect) - 1] = (x, y + 1)
			if k == 81: 		#LeftKey
				_PricerImageRect[len(_PricerImageRect) - 1] = (x - 1, y)
			if k == 83: 		#RightKey
				_PricerImageRect[len(_PricerImageRect) - 1] = (x + 1, y)

	cords_string = ''
	for x, y in _PricerImageRect:
		cords_string = ' '.join([cords_string, str(x), str(y)])
	out_string = ' '.join([image_name, cords_string, '\n'])
	with open(output_file, 'a') as outfile:
		outfile.write(out_string)
	cv2.destroyAllWindows()
	return 0

if __name__ == '__main__':
	sharps = '#################################'
	TIP = sharps + '\nUse:\nLeftMouseButton to add point\nUp/Down/Left/Right buttons to change last point position\nBackspace to delete last point\nEnter to save 4 points on current image\np to print current coordinates\nEscape to exit\n' + sharps
	OUT_FILE_NAME = os.environ['BEORGDATA'] + '/ToMark/rubli.txt'
	DATA_PATH = os.environ['BEORGDATA'] + '/ToMark'

	out_file = open(os.path.join(DATA_PATH, OUT_FILE_NAME), 'w')
	out_file.close()

	files = os.listdir(DATA_PATH)
	img_list = filter(lambda x: x.endswith('.jpg'), files)

	print TIP
	for img in img_list:
		if (askPoints(4, DATA_PATH, img, OUT_FILE_NAME) == -1):
			sys.exit()
