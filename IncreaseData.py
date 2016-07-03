import cv2
import numpy as np
import random

frr = open('SData/rubli.txt', 'r')
frk = open('SData/kopeiki.txt', 'r')
frn = open('SData/name.txt', 'r')
fwr = open('LData/rubli.txt', 'w')
fwk = open('LData/kopeiki.txt', 'w')
fwn = open('LData/name.txt', 'w')

k = 25

sr = frr.readline().split()
sk = frk.readline().split()
sn = frn.readline().split()

while(len(sr) != 0):
	print sr[0]
	
	pr1, pr2, pr3, pr4 = sr[1:3], sr[3:5], sr[5:7], sr[7:9]
	pr1 = [int(pr1[0]), int(pr1[1])]
	pr2 = [int(pr2[0]), int(pr2[1])]
	pr3 = [int(pr3[0]), int(pr3[1])]
	pr4 = [int(pr4[0]), int(pr4[1])]

	pk1, pk2, pk3, pk4 = sk[1:3], sk[3:5], sk[5:7], sk[7:9]
	pk1 = [int(pk1[0]), int(pk1[1])]
	pk2 = [int(pk2[0]), int(pk2[1])]
	pk3 = [int(pk3[0]), int(pk3[1])]
	pk4 = [int(pk4[0]), int(pk4[1])]

	pn1, pn2, pn3, pn4 = sn[1:3], sn[3:5], sn[5:7], sn[7:9]
	pn1 = [int(pn1[0]), int(pn1[1])]
	pn2 = [int(pn2[0]), int(pn2[1])]
	pn3 = [int(pn3[0]), int(pn3[1])]
	pn4 = [int(pn4[0]), int(pn4[1])]
	
	for i in range(k):

		ppr1 = pr1
		ppr2 = pr2
		ppr3 = pr3
		ppr4 = pr4
		
		ppk1 = pk1
		ppk2 = pk2
		ppk3 = pk3
		ppk4 = pk4
		
		ppn1 = pn1
		ppn2 = pn2
		ppn3 = pn3
		ppn4 = pn4
		
		img = cv2.imread("./SData/" + sr[0])
		rows,cols, ch  = img.shape

		###########BLURING
		ker1 = int (random.random() * 15) + 1
		ker2 = int (random.random() * 15) + 1
		img = cv2.blur(img,(ker1, ker2))
		##################

		############COLORS
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		hsv[:,:,0] = hsv[:,:,0] + int(40 * (random.random() - 0.5))
		img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
		##################

		#############GAMMA
		gamma = random.random() * 2.5 + 0.25
		invGamma = 1.0 / gamma
		table = np.array([((j / 255.0) ** invGamma) * 255
			for j in np.arange(0, 256)]).astype("uint8")
		img = cv2.LUT(img, table)
		##################

		##########ROTATION
		deg = 5
		deg *= 2 * (random.random() - 0.5)
		M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
		
		ppr1 = [ppr1[0] * M[0][0] + ppr1[1] * M[0][1] + M[0][2], ppr1[0] * M[1][0] + ppr1[1] * M[1][1] + M[1][2]]
		ppr2 = [ppr2[0] * M[0][0] + ppr2[1] * M[0][1] + M[0][2], ppr2[0] * M[1][0] + ppr2[1] * M[1][1] + M[1][2]]
		ppr3 = [ppr3[0] * M[0][0] + ppr3[1] * M[0][1] + M[0][2], ppr3[0] * M[1][0] + ppr3[1] * M[1][1] + M[1][2]]
		ppr4 = [ppr4[0] * M[0][0] + ppr4[1] * M[0][1] + M[0][2], ppr4[0] * M[1][0] + ppr4[1] * M[1][1] + M[1][2]]
		
		ppk1 = [ppk1[0] * M[0][0] + ppk1[1] * M[0][1] + M[0][2], ppk1[0] * M[1][0] + ppk1[1] * M[1][1] + M[1][2]]
		ppk2 = [ppk2[0] * M[0][0] + ppk2[1] * M[0][1] + M[0][2], ppk2[0] * M[1][0] + ppk2[1] * M[1][1] + M[1][2]]
		ppk3 = [ppk3[0] * M[0][0] + ppk3[1] * M[0][1] + M[0][2], ppk3[0] * M[1][0] + ppk3[1] * M[1][1] + M[1][2]]
		ppk4 = [ppk4[0] * M[0][0] + ppk4[1] * M[0][1] + M[0][2], ppk4[0] * M[1][0] + ppk4[1] * M[1][1] + M[1][2]]
		
		ppn1 = [ppn1[0] * M[0][0] + ppn1[1] * M[0][1] + M[0][2], ppn1[0] * M[1][0] + ppn1[1] * M[1][1] + M[1][2]]
		ppn2 = [ppn2[0] * M[0][0] + ppn2[1] * M[0][1] + M[0][2], ppn2[0] * M[1][0] + ppn2[1] * M[1][1] + M[1][2]]
		ppn3 = [ppn3[0] * M[0][0] + ppn3[1] * M[0][1] + M[0][2], ppn3[0] * M[1][0] + ppn3[1] * M[1][1] + M[1][2]]
		ppn4 = [ppn4[0] * M[0][0] + ppn4[1] * M[0][1] + M[0][2], ppn4[0] * M[1][0] + ppn4[1] * M[1][1] + M[1][2]]
		img = cv2.warpAffine(img,M,(cols,rows))
		##################

		#######TRANSLATION 
		#dx = (random.random() - 1) * cols * 0.05
		#dy = (random.random() - 1) * rows * 0.2
		#pp1 += [dx, dy]
		#pp2 += [dx, dy]
		#pp3 += [dx, dy]
		#pp4 += [dx, dy]
		#M = np.float32([[1, 0, dx], [0, 1, dy]])
		#img = cv2.warpAffine(img,M,(cols,rows))
		##################

		#######PERSPECTIVE
		deg = 5
		x1, y1  = random.random() * deg, random.random() * deg
		x2, y2  = random.random() * deg, random.random() * deg
		x3, y3  = random.random() * deg, random.random() * deg
		x4, y4  = random.random() * deg, random.random() * deg
		pts1 = np.float32([[x1, y1],[300 - x2, y2],[x3,300 - y3],[300 - x4,300 - y4]])
		pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
		M = cv2.getPerspectiveTransform(pts1,pts2)
		
		tr1 = M[2][0] * ppr1[0] + M[2][1] * ppr1[1] + M[2][2]
		ppr1 = [(ppr1[0] * M[0][0] + ppr1[1] * M[0][1] + M[0][2]) / tr1, (ppr1[0] * M[1][0] + ppr1[1] * M[1][1] + M[1][2]) / tr1]
		tr2 = M[2][0] * ppr2[0] + M[2][1] * ppr2[1] + M[2][2]
		ppr2 = [(ppr2[0] * M[0][0] + ppr2[1] * M[0][1] + M[0][2]) / tr2, (ppr2[0] * M[1][0] + ppr2[1] * M[1][1] + M[1][2]) / tr2]
		tr3 = M[2][0] * ppr3[0] + M[2][1] * ppr3[1] + M[2][2]
		ppr3 = [(ppr3[0] * M[0][0] + ppr3[1] * M[0][1] + M[0][2]) / tr3, (ppr3[0] * M[1][0] + ppr3[1] * M[1][1] + M[1][2]) / tr3]
		tr4 = M[2][0] * ppr4[0] + M[2][1] * ppr4[1] + M[2][2]
		ppr4 = [(ppr4[0] * M[0][0] + ppr4[1] * M[0][1] + M[0][2]) / tr4, (ppr4[0] * M[1][0] + ppr4[1] * M[1][1] + M[1][2]) / tr4]
		
		tk1 = M[2][0] * ppk1[0] + M[2][1] * ppk1[1] + M[2][2]
		ppk1 = [(ppk1[0] * M[0][0] + ppk1[1] * M[0][1] + M[0][2]) / tk1, (ppk1[0] * M[1][0] + ppk1[1] * M[1][1] + M[1][2]) / tk1]
		tk2 = M[2][0] * ppk2[0] + M[2][1] * ppk2[1] + M[2][2]
		ppk2 = [(ppk2[0] * M[0][0] + ppk2[1] * M[0][1] + M[0][2]) / tk2, (ppk2[0] * M[1][0] + ppk2[1] * M[1][1] + M[1][2]) / tk2]
		tk3 = M[2][0] * ppk3[0] + M[2][1] * ppk3[1] + M[2][2]
		ppk3 = [(ppk3[0] * M[0][0] + ppk3[1] * M[0][1] + M[0][2]) / tk3, (ppk3[0] * M[1][0] + ppk3[1] * M[1][1] + M[1][2]) / tk3]
		tk4 = M[2][0] * ppk4[0] + M[2][1] * ppk4[1] + M[2][2]
		ppk4 = [(ppk4[0] * M[0][0] + ppk4[1] * M[0][1] + M[0][2]) / tk4, (ppk4[0] * M[1][0] + ppk4[1] * M[1][1] + M[1][2]) / tk4]
		
		tn1 = M[2][0] * ppn1[0] + M[2][1] * ppn1[1] + M[2][2]
		ppn1 = [(ppn1[0] * M[0][0] + ppn1[1] * M[0][1] + M[0][2]) / tn1, (ppn1[0] * M[1][0] + ppn1[1] * M[1][1] + M[1][2]) / tn1]
		tn2 = M[2][0] * ppn2[0] + M[2][1] * ppn2[1] + M[2][2]
		ppn2 = [(ppn2[0] * M[0][0] + ppn2[1] * M[0][1] + M[0][2]) / tn2, (ppn2[0] * M[1][0] + ppn2[1] * M[1][1] + M[1][2]) / tn2]
		tn3 = M[2][0] * ppn3[0] + M[2][1] * ppn3[1] + M[2][2]
		ppn3 = [(ppn3[0] * M[0][0] + ppn3[1] * M[0][1] + M[0][2]) / tn3, (ppn3[0] * M[1][0] + ppn3[1] * M[1][1] + M[1][2]) / tn3]
		tn4 = M[2][0] * ppn4[0] + M[2][1] * ppn4[1] + M[2][2]
		ppn4 = [(ppn4[0] * M[0][0] + ppn4[1] * M[0][1] + M[0][2]) / tn4, (ppn4[0] * M[1][0] + ppn4[1] * M[1][1] + M[1][2]) / tn4]
		
		img = cv2.warpPerspective(img,M,(cols, rows))
		##################

		ppr1 = [str(int(ppr1[0])), str(int(ppr1[1]))]
		ppr2 = [str(int(ppr2[0])), str(int(ppr2[1]))]
		ppr3 = [str(int(ppr3[0])), str(int(ppr3[1]))]
		ppr4 = [str(int(ppr4[0])), str(int(ppr4[1]))]

		ppk1 = [str(int(ppk1[0])), str(int(ppk1[1]))]
		ppk2 = [str(int(ppk2[0])), str(int(ppk2[1]))]
		ppk3 = [str(int(ppk3[0])), str(int(ppk3[1]))]
		ppk4 = [str(int(ppk4[0])), str(int(ppk4[1]))]

		ppn1 = [str(int(ppn1[0])), str(int(ppn1[1]))]
		ppn2 = [str(int(ppn2[0])), str(int(ppn2[1]))]
		ppn3 = [str(int(ppn3[0])), str(int(ppn3[1]))]
		ppn4 = [str(int(ppn4[0])), str(int(ppn4[1]))]
		
		sp = ' '
		fwr.write(sr[0][0: len(sr[0]) - 4] + '_' + str(i) + '.jpg' + ' ' + sp.join(ppr1 + ppr2 + ppr3 + ppr4) + '\n')
		fwk.write(sr[0][0: len(sr[0]) - 4] + '_' + str(i) + '.jpg' + ' ' + sp.join(ppk1 + ppk2 + ppk3 + ppk4) + '\n')
		fwn.write(sr[0][0: len(sr[0]) - 4] + '_' + str(i) + '.jpg' + ' ' + sp.join(ppn1 + ppn2 + ppn3 + ppn4) + '\n')
		cv2.imwrite('./LData/' + sr[0][0: len(sr[0]) - 4] + '_' + str(i) + '.jpg',img)

	sr = frr.readline().split()
	sk = frk.readline().split()
	sn = frn.readline().split()

frr.close()
frk.close()
frn.close()
fwr.close()
fwk.close()
fwn.close()
