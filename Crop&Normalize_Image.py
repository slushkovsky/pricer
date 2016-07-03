import cv2
import numpy as np

f = open('CData/rubli.txt', 'r')
s = f.readline().split()
while(len(s) != 0):
	print s[0]
	img =  cv2.imread("./CData/" + s[0])
	rows,cols, ch  = img.shape

	p1, p2, p3, p4 = s[1:3], s[3:5], s[5:7], s[7:9]
	pts1 = np.float32([p1, p2, p4, p3])
	pts2 = np.float32([[0,0],[250,0],[0,250],[500,250]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	img = cv2.warpPerspective(img,M,(cols, rows))

	img = img[0:250, 0:500]
	
	cv2.imwrite('./CData/' + s[0] ,img)
	s = f.readline().split()
