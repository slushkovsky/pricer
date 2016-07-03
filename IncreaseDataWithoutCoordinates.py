import cv2
import numpy as np
from os import listdir
import random

files = listdir("./SData")
jpg = filter(lambda x: x.endswith('.jpg'), files)

k = 20

for img_name in jpg:
	print img_name
	for i in range(k):
		img = cv2.imread("./SData/" + img_name)
		rows,cols, ch  = img.shape

		#BLURING
		ker1 = int (random.random() * 50) + 1
		ker2 = int (random.random() * 50) + 1
		img = cv2.blur(img,(ker1, ker2))

		#COLORS
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		hsv[:,:,0] = hsv[:,:,0] + int(40 * (random.random() - 0.5))
		img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

		#GAMMA
		gamma = random.random() * 2.5 + 0.25
		invGamma = 1.0 / gamma
		table = np.array([((j / 255.0) ** invGamma) * 255
			for j in np.arange(0, 256)]).astype("uint8")
		img = cv2.LUT(img, table)

		#ROTATION
		deg = (random.random() - 0.5) * 12
		M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
		img = cv2.warpAffine(img,M,(cols,rows))

		#TRANSLATION 
		dx = (random.random() - 1) * cols * 0.05
		dy = (random.random() - 1) * rows * 0.2
		M = np.float32([[1, 0, dx], [0, 1, dy]])
		img = cv2.warpAffine(img,M,(cols,rows))

		#PERSPECTIVE
		x1, y1  = (random.random() - 0.5) * 6, (random.random() - 0.5) * 6
		x2, y2  = (random.random() - 0.5) * 6, (random.random() - 0.5) * 6
		x3, y3  = (random.random() - 0.5) * 6, (random.random() - 0.5) * 6
		x4, y4  = (random.random() - 0.5) * 6, (random.random() - 0.5) * 6
		pts1 = np.float32([[x1, y1],[300 + x2, y2],[x3,300 + y3],[300 + x4,300 + y4]])
		pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
		M = cv2.getPerspectiveTransform(pts1,pts2)
		img = cv2.warpPerspective(img,M,(cols, rows))

		cv2.imwrite('./LData/' + img_name[0: len(img_name) - 4] + '_' + str(i) + '.jpg',img)

