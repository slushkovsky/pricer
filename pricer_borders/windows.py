from utils.sci_window import SciWindow
from settings import CANNY_PARAMS, HOUGH_PARAMS, DILATE_PARAMS, TRESHOLD_PARAMS

class CannyWindow(SciWindow): 
	window_name = 'canny'
	defaults = CANNY_PARAMS

	trackbars = {
		'T1': 'thresh_1', 
		'T2': 'thresh_2'
	}

	trackbar_max = {
		'T1': 255,
		'T2': 255
	}
 
 
class TresholdWindow(SciWindow): 
	window_name = 'treshold'
	defaults = TRESHOLD_PARAMS

	trackbars = {
              'p1': 'p1',
              'p2': 'p2'
	}

	trackbar_max = {
                 'p1': 255,
                 'p2': 255
	}
 
 
class DilateWindow(SciWindow): 
	window_name = 'dilate'
	defaults = DILATE_PARAMS

	trackbars = {
		'iter': 'iter', 
	}

	trackbar_max = {
		'iter': 10,
	}


class HoughWindow(SciWindow): 
	window_name = 'hough'
	defaults = HOUGH_PARAMS

	trackbars = {
		'MIN_LEN': 'min_len', 
		'MAX_GAP': 'max_gap', 
		'THRESH': 'thresh'
	}

	trackbar_max = {
		'MIN_LEN': 1000,
		'MAX_GAP': 100, 
		'THRESH': 1000
	}