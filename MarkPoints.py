import cv2
import numpy as np
import shutil
import os
import sys

#BE CAREFULL: GLOBAL LIST!
_PricerImageRect = []
_PricerCurChanged = 0

def rectCallback(event, x, y, flags, param):
	global _PricerImageRect
	global _PricerCurChanged
	if event == cv2.EVENT_LBUTTONUP:
		_PricerImageRect.append((x, y))
		_PricerCurChanged = 0

def askPoints(numPoints, data_path, image_name, output_file, suf = ''):
	global _PricerImageRect
	global _PricerCurChanged
	image_path = os.path.join(data_path, image_name)
	_PricerImageRect = []
	window_name = 'Mark ' + str(numPoints) + ' points' + suf
	cv2.namedWindow(window_name)
	cv2.setMouseCallback(window_name, rectCallback)
	cur = 0
	_PricerCurChanged = 0
	while(True):
		image = cv2.imread(image_path)
		size = len(_PricerImageRect)
		for i in range(size):
			cv2.circle(image, _PricerImageRect[i], 5, (0, 255, 0), -1)
			if (i == cur):
				cv2.circle(image, _PricerImageRect[i], 5, (180, 255, 180), -1)
			cv2.line(image, _PricerImageRect[i], _PricerImageRect[(i + 1) % size], (0, 255, 0))
		cv2.imshow(window_name, image)
		if _PricerCurChanged == 0:
			cur = len(_PricerImageRect) - 1
		k = cv2.waitKey(100) & 0xFF
		if k == 27:				#ESC
			return -1
		if k == ord('p'):		#Print coords
			print _PricerImageRect
		if len(_PricerImageRect) == numPoints:
			if k == 10:			#Enter
				break
		if k == ord('s'):
			cv2.destroyAllWindows()
			return 0
		if k >= ord('1') and k <= ord('9') and (k - ord('1')) < len(_PricerImageRect):
			cur = k - ord('1')
			_PricerCurChanged = 1
		if(len(_PricerImageRect) != 0):
			x, y = _PricerImageRect[cur]
			if k == 82: 		#UpKey
				_PricerImageRect[cur] = (x, y - 1)
			if k == 84: 		#DownKey
				_PricerImageRect[cur] = (x, y + 1)
			if k == 81: 		#LeftKey
				_PricerImageRect[cur] = (x - 1, y)
			if k == 83: 		#RightKey
				_PricerImageRect[cur] = (x + 1, y)
			if k == 8:	#Backspace
				del _PricerImageRect[len(_PricerImageRect) - 1]

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
	TIP = sharps + '\nCall this script whis parametr new to clear output file\nUsage:\nLeftMouseButton to add point\nUp/Down/Left/Right buttons to change current point position\n1/2/3/4 to change current point\ns to skip this image\nBackspace to delete last point\nEnter to save 4 points on current image\np to print current coordinates\nEscape to exit\n' + sharps
	OUT_FILE_NAME = os.environ['BEORGDATA'] + '/ToMark/kopeiki.txt'
	DATA_PATH = os.environ['BEORGDATA'] + '/ToMark'

	if len(sys.argv) > 1 and sys.argv[1] == 'new':
		out_file = open(os.path.join(DATA_PATH, OUT_FILE_NAME), 'w')
		out_file.close()

	files = os.listdir(DATA_PATH)
	img_list = filter(lambda x: x.endswith('.jpg'), files)

	print TIP
	out_file = open(os.path.join(DATA_PATH, OUT_FILE_NAME), 'r')
	size = len(out_file.readlines())
	if len(sys.argv) > 1 and sys.argv[1] == 'new':
		size = -1
	i = 0
	out_file.close()
	for i in range(len(img_list)):
		if i < size:
			continue
		elif askPoints(4, DATA_PATH, img_list[i], OUT_FILE_NAME, ' ( ' + str(i) + ' )') == -1:
			sys.exit()
