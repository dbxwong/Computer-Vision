import numpy as np
import cv2
from matchPics import matchPics
from scipy import ndimage, misc
import matplotlib.pyplot as plt
#from helper import briefMatch
from helper import computeBrief
#from helper import corner_detection
from helper import plotMatches
from opts import get_opts

def rotTest():
	opts = get_opts()
	ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
	sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'

	#Q2.1.6
	#Read the image and convert to grayscale
	cv_cover = cv2.imread('../data/cv_cover.jpg')
	img = cv_cover
	
	locs = [] # N x 2 matrix containing x,y coords or matched point pairs
	hist = []
	num_matches=[]
	bin_list = []
	for i in range(36):
		print (i)
		#Rotate Image
		rotImg = ndimage.rotate(img, i*10, reshape=True)
		
		#Compute features, descriptors and Match features
		img_matches, locs1, locs2 = matchPics(rotImg, img, opts)
		#plotMatches(rotImg, img, img_matches, locs1, locs2) # display matches between both pictures
		num_matches.append(len(img_matches))
		print (len(img_matches))
		#plt.hist(num_matches, bins=36, range=None, density=False) ## put shape of matches in histogram
		plt.bar(i*10, height=num_matches[i]) ## put shape of matches in histogram
		
		plt.title('Histogram of matches')
		plt.ylabel('Frequency')
		plt.xlabel('Number of matches')
		
						
	#Display histogram
	#plt.title('Histogram of matches')
	#plt.ylabel('Frequency')
	#plt.xlabel('Number of matches')
	plt.show()

	return 
