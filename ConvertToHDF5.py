from os import environ
import shutil

import cv2
import numpy as np
import h5py

#########################
#IT MAY NOT WORK PROPERLY
#########################

def generateHDF5(data_path, file_name, out_name, image_size, n_chan = 3, dataset_data_name = 'data', dataset_label_name = 'label'):
	file_path = '/'.join([data_path, file_name])
	infile = open(file_path, 'r')
	num_lines = sum(1 for line in infile.readlines())
	infile.seek(0)

	data = np.zeros((num_lines, n_chan, image_size[0], image_size[1]), dtype="f4")
	label = np.zeros((num_lines, 8), dtype='f4')

	with h5py.File(out_name, 'w') as out:
		i = 0
		for line in infile.readlines():
			img_name, tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = line.split()

			img_path = '/'.join([data_path, img_name])
			img = cv2.imread(img_path)
			size = img.shape
			img = cv2.resize(img, image_size, interpolation = cv2.INTER_CUBIC)
			img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
			img = img.transpose((2,1,0))

			contour = [int(tl_x) / size[0], int(tl_y) / size[1], 
			int(tr_x) / size[0], int(tr_y) / size[1], int(br_x) / size[0], 
			int(br_y) / size[1], int(bl_x) / size[0], int(bl_y) / size[1]]

			data[i] = img
			label[i] = contour
			i += 1

		out.create_dataset(dataset_data_name, data=data)
		out.create_dataset(dataset_label_name, data=label)
		out.close()

	infile.close()

	lst_file = open(out_name + '.list.txt','w')
	lst_file.write(out_name)
	lst_file.close()

if __name__ == '__main__':
	DATA_PATH = environ['BEORGDATAGEN'] + '/FinalData'
	generateHDF5(DATA_PATH, 'rubli.txt', 'rubliTrain60x30.hdf5', (60, 30))
