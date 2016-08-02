from collections import namedtuple

CannyParams = namedtuple('CannyParams', ['thresh_1', 'thresh_2'])
HoughParams = namedtuple('HoughParams', ['min_len', 'max_gap', 'thresh'])
DilateParams = namedtuple('DilateParams', ['iter'])
TresholdParams = namedtuple('TresholdParams', ['p1', 'p2'])

CANNY_PARAMS = CannyParams(50, 150)
HOUGH_PARAMS = HoughParams(100, 5, 106)
DILATE_PARAMS = DilateParams(1)
TRESHOLD_PARAMS = TresholdParams(131, 2)