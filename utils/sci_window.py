import cv2

class SciWindow(object):
	window_name = ''
	trackbars = {}

	def __init__(self): 
		cv2.namedWindow(self.window_name)

		for name, field in self.trackbars.items(): 
			cv2.createTrackbar(name, self.window_name, 
				               getattr(self.defaults, field), 
			                   self.trackbar_max[name],
			                   lambda x: None)

	def get(self): 
		kw = {}

		for name, field in self.trackbars.items():
			kw[field] = cv2.getTrackbarPos(name, self.window_name)

		return type(self.defaults)(**kw)

	def show(self, img): 
		cv2.imshow(self.window_name, img)
