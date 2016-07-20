import cv2
import numpy as np
import shutil
import os
import sys

def correctPoints(rect, data_path, image_name, output_file, suf = ''):
	image_path = os.path.join(data_path, image_name)
	window_name = 'Correct ' + str(len(rect)) + ' points' + suf
	cv2.namedWindow(window_name)
	cur = 0
	while(True):
		image = cv2.imread(image_path)
		for i in range(len(rect)):
			cv2.circle(image, rect[i], 5, (0, 255, 0), -1)
			cv2.line(image, rect[i], rect[(i + 1) % len(rect)], (0, 255, 0))
		cv2.imshow(window_name, image)
		k = cv2.waitKey(100) & 0xFF
		if k == 27:				#ESC
			return -1
		if k == ord('p'):		#Print coords
			print rect
		if k == 10:				#Enter
			break
		if k == ord('s'):
			cv2.destroyAllWindows()
			return 0
		if k >= ord('1') and k <= ord('9') and (k - ord('1')) < len(rect):
			cur = k - ord('1')
		x, y = rect[cur]
		if k == 82: 		#UpKey
			rect[cur] = (x, y - 1)
		if k == 84: 		#DownKey
			rect[cur] = (x, y + 1)
		if k == 81: 		#LeftKey
			rect[cur] = (x - 1, y)
		if k == 83: 		#RightKey
			rect[cur] = (x + 1, y)

	cords_string = ''
	for x, y in rect:
		cords_string = ' '.join([cords_string, str(x), str(y)])
	out_string = ' '.join([image_name, cords_string, '\n'])
	with open(output_file, 'a') as outfile:
		outfile.write(out_string)
	cv2.destroyAllWindows()
	return 0

if __name__ == '__main__':
	sharps = '#################################'
	TIP = sharps + '\nCall this script whis parametr new to clear output file\nUsage:\nUp/Down/Left/Right buttons to change current point position\n1/2/3/4 to change current point\ns to skip this image\nEnter to save image and go to the next\np to print current coordinates\nEscape to exit\n' + sharps
	OUT_FILE_NAME = os.environ['BEORGDATA'] + '/ToMark/rubli_corrected.txt'
	IN_FILE_NAME = os.environ['BEORGDATA'] + '/ToMark/rubli.txt'
	DATA_PATH = os.environ['BEORGDATA'] + '/ToMark'

	in_file = open(IN_FILE_NAME, 'r')
	lines = in_file.readlines()
	in_size = len(lines)

	print TIP

	if len(sys.argv) > 1 and sys.argv[1] == 'new':
		out_file = open(os.path.join(DATA_PATH, OUT_FILE_NAME), 'w')
		out_file.close()

	out_file = open(os.path.join(DATA_PATH, OUT_FILE_NAME), 'r')
	out_size = len(out_file.readlines())
	out_file.close()

	for i in range(in_size):
		img_name, tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = lines[i]	.split()
		coords = [(int(tl_x), int(tl_y)), (int(tr_x), int(tr_y)), (int(br_x), int(br_y)), (int(bl_x), int(bl_y))]
		if i < out_size:
			continue
		elif correctPoints(coords, DATA_PATH, img_name, OUT_FILE_NAME, ' ( ' + str(i) + '/' + str(in_size) + ' )') == -1:
			sys.exit()

