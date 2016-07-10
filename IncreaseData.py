import cv2
import numpy as np
import random
import shutil
from os import listdir
from os import path, makedirs, environ

def Blur(param, img):
	ker1 = int (random.random() * param) + 1
	ker2 = int (random.random() * param) + 1
	return cv2.blur(img,(ker1, ker2))

def ChangeColors(param, img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hsv[:,:,0] = hsv[:,:,0] + int(param * (random.random() - 0.5))
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def Gamma(param, img):
	gamma = random.random() * param + 0.25
	invGamma = 1.0 / gamma
	table = np.array([((j / 255.0) ** invGamma) * 255
		for j in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(img, table)

def RotateMatrix(param, matrix, img):
	rows, cols, ch  = img.shape
	M = cv2.getRotationMatrix2D((cols/2, rows/2), param, 1)
	lst = list(np.dot(M, matrix))
	lst.append([1, 1, 1, 1])
	return lst

def RotateImg(param, img):
	rows, cols, ch  = img.shape
	deg = param * 2 * (random.random() - 0.5)
	M = cv2.getRotationMatrix2D((cols/2, rows/2), deg, 1)
	img = cv2.warpAffine(img,M,(cols,rows))
	return img, deg

def TranslateMatrix(param, matrix):
	dx, dy = param
	M = np.float32([[1, 0, dx], [0, 1, dy]])
	lst = list(np.dot(M, matrix))
	lst.append([1, 1, 1, 1])
	return lst

def TranslateImg(param, img):
	max_dx, max_dy = param
	rows, cols, ch  = img.shape
	dx = (random.random() - 0.5) * cols * max_dx
	dy = (random.random() - 0.5) * rows * max_dy
	M = np.float32([[1, 0, dx], [0, 1, dy]])
	img = cv2.warpAffine(img,M,(cols,rows))
	return img, (dx, dy)

def PerspectiveMatrix(param, matrix, img):
	rows, cols, ch  = img.shape
	x1, y1, x2, y2, x3, y3, x4, y4 = param

	pts1 = np.float32([[x1, y1],[300 - x2, y2],[x3,300 - y3],[300 - x4,300 - y4]])
	pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
	M = cv2.getPerspectiveTransform(pts1,pts2)

	matrix = np.dot(M, matrix)
	for j in range(4):
		for i in range(3):
			matrix[i][j] /= matrix[2][j]
	return matrix

def PerspectiveImg(param, img):
	rows, cols, ch  = img.shape
	x1, y1  = random.random() * param, random.random() * param
	x2, y2  = random.random() * param, random.random() * param
	x3, y3  = random.random() * param, random.random() * param
	x4, y4  = random.random() * param, random.random() * param
	pts1 = np.float32([[x1, y1],[300 - x2, y2],[x3,300 - y3],[300 - x4,300 - y4]])
	pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	img = cv2.warpPerspective(img, M, (cols, rows))
	return img, (x1, y1, x2, y2, x3, y3, x4, y4)

def GetCords(lst):
	ans =  [[int(lst[1])], [int(lst[2])], [1]]
	for i in range(3):
		ans[i] += [[int(lst[3])], [int(lst[4])], [1]][i]
		ans[i] += [[int(lst[5])], [int(lst[6])], [1]][i] 
		ans[i] += [[int(lst[7])], [int(lst[8])], [1]][i]
	return ans
	#return [[int(lst[1]), int(lst[2]), 1] , [int(lst[3]), int(lst[4]), 1], [int(lst[5]), int(lst[6]), 1], [int(lst[7]), int(lst[8]), 1]]

def WriteMatrix(matrix, new_img_name, outfile):
	outfile.write(new_img_name + ' ')
	lst = []
	for i in range(4):
		lst += [matrix[0][i], matrix[1][i]]
	str_lst = [str(int(i)) for i in lst]
	outfile.write(' '.join(str_lst) + '\n')

def IncreaseDataPricer(repeats, data_path, rubl_file_name, kope_file_name, name_file_name, out_path):
	if(path.exists(out_path)):
		ans = raw_input('Path already exist. Would you like delete it? y/n \n')
		if ans == 'y':
			shutil.rmtree(out_path)
		else:
			print 'Specify a different path'
			return
	makedirs(out_path)

	rubl_file_path = '/'.join([data_path, rubl_file_name])
	kope_file_path = '/'.join([data_path, kope_file_name])
	name_file_path = '/'.join([data_path, name_file_name])
	rubl_outfile_path = '/'.join([out_path, rubl_file_name])
	kope_outfile_path = '/'.join([out_path, kope_file_name])
	name_outfile_path = '/'.join([out_path, name_file_name])

	rubl_infile = open(rubl_file_path, 'r')
	kope_infile = open(kope_file_path, 'r')
	name_infile = open(name_file_path, 'r')
	rubl_outfile = open(rubl_outfile_path, 'w')
	kope_outfile = open(kope_outfile_path, 'w')
	name_outfile = open(name_outfile_path, 'w')

	inpR = rubl_infile.readline().split()
	inpK = kope_infile.readline().split()
	inpN = name_infile.readline().split()

	while(len(inpR) != 0 and len(inpK) != 0 and len(inpN) != 0):
		RublMatrix = GetCords(inpR)
		KopeMatrix = GetCords(inpK)
		NameMatrix = GetCords(inpN)
		print inpR[0]
		for i in range(repeats):
			RublMatrix_i = RublMatrix
			KopeMatrix_i = KopeMatrix
			NameMatrix_i = NameMatrix

			if (inpR[0] != inpK[0] or inpR[0] != inpN[0]):
				print 'Local files contradict each other'
				return -1
			img_name = inpR[0]
			img_path = '/'.join([data_path, img_name])
			img = cv2.imread(img_path)

			img = Blur(15, img)
			img = ChangeColors(40, img)
			img = Gamma(2.5, img)

			img, deg = RotateImg(6, img)
			RublMatrix_i = RotateMatrix(deg, RublMatrix_i, img)
			KopeMatrix_i = RotateMatrix(deg, KopeMatrix_i, img)
			NameMatrix_i = RotateMatrix(deg, NameMatrix_i, img)

			img, tr = TranslateImg((0.05, 0.05), img)
			RublMatrix_i = TranslateMatrix(tr, RublMatrix_i)
			KopeMatrix_i = TranslateMatrix(tr, KopeMatrix_i)
			NameMatrix_i = TranslateMatrix(tr, NameMatrix_i)

			img, pr = PerspectiveImg(25, img)
			RublMatrix_i = PerspectiveMatrix(pr, RublMatrix_i, img)
			KopeMatrix_i = PerspectiveMatrix(pr, KopeMatrix_i, img)
			NameMatrix_i = PerspectiveMatrix(pr, NameMatrix_i, img)

			new_img_name = img_name[0: len(img_name) - 4] + '_' + str(i) + '.jpg'
			new_img_path = '/'.join([out_path, new_img_name])
			WriteMatrix(RublMatrix_i, new_img_name, rubl_outfile)
			WriteMatrix(KopeMatrix_i, new_img_name, kope_outfile)
			WriteMatrix(NameMatrix_i, new_img_name, name_outfile)
			cv2.imwrite(new_img_path, img)

		inpR = rubl_infile.readline().split()
		inpK = kope_infile.readline().split()
		inpN = name_infile.readline().split()

	rubl_infile.close()
	kope_infile.close()
	name_infile.close()
	rubl_outfile.close()
	kope_outfile.close()
	name_outfile.close()

def IncreaseDataCords(repeats, data_path, file_name, out_path):
	if(path.exists(out_path)):
		ans = raw_input('Path already exist. Would you like delete it? y/n \n')
		if ans == 'y':
			shutil.rmtree(out_path)
		else:
			print 'Specify a different path'
			return
	makedirs(out_path)
	file_path = '/'.join([data_path, file_name])
	outfile_path = '/'.join([out_path, file_name])
	infile = open(file_path, 'r')
	outfile = open(outfile_path, 'w')
	inp = infile.readline().split()

	while(len(inp) != 0):
		matrix = GetCords(inp)
		for i in range(repeats):
			matrix_i = matrix
			img_name = inp[0]
			img_path = '/'.join([data_path, img_name])
			img = cv2.imread(img_path)

			img = Blur(15, img)
			img = ChangeColors(40, img)
			img = Gamma(2.5, img)
			img, deg = RotateImg(5, img)
			matrix_i = RotateMatrix(deg, matrix, img)
			img, tr = TranslateImg((0.05, 0.005), img)
			matrix_i = TranslateMatrix(tr, matrix)
			img, pr = PerspectiveImg(25, img)
			matrix_i = PerspectiveMatrix(pr, matrix, img)

			new_img_name = img_name[0: len(img_name) - 4] + '_' + str(i) + '.jpg'
			new_img_path = '/'.join([out_path, new_img_name])
			WriteMatrix(matrix_i, new_img_name, outfile)
			cv2.imwrite(new_img_path, img)
		inp = infile.readline().split()

	infile.close()
	outfile.close()


def IncreaseData(repeats, data_path, out_path):
	if(path.exists(out_path)):
		ans = raw_input('Path already exist. Would you like delete it? y/n \n')
		if ans == 'y':
			shutil.rmtree(out_path)
		else:
			print 'Specify a different path'
			return
	makedirs(out_path)

	files = listdir(data_path)
	jpg = filter(lambda x: x.endswith('.jpg'), files)
	for img_name in jpg:
		print img_name
		img_path = '/'.join([data_path, img_name]) 
		for i in range(repeats):
			img = cv2.imread(img_path)
			img = Blur(15, img)
			img = ChangeColors(40, img)
			img = Gamma(2.5, img)
			img, deg = RotateImg(5, img)
			img, tr = TranslateImg((0.05, 0.05), img)
			img, pr = PerspectiveImg(25, img)
			new_img_name = img_name[0: len(img_name) - 4] + '_' + str(i) + '.jpg'
			new_img_path = '/'.join([out_path, new_img_name])
			cv2.imwrite(new_img_path, img)

if __name__ == '__main__':
	DATA_PATH = environ['BEORGDATAGEN'] + '/CropedImages'
	OUT_PATH = environ['BEORGDATAGEN'] + '/FinalData'
	IncreaseDataPricer(25, DATA_PATH, 'rubli.txt', 'kopeiki.txt', 'name.txt', OUT_PATH)
