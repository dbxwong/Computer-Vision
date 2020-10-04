
import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
from planarH import computeH

opts = get_opts()

#load images
#img1 = cv2.imread('../data/pano_left.jpg')
#img2 = cv2.imread('../data/pano_right.jpg') 

img1 = cv2.imread('D:/MRSD/0.4 Computer Vision (16-720AC)/HW2/data/pano_left.jpg')
img2 = cv2.imread('D:/MRSD/0.4 Computer Vision (16-720AC)/HW2/data/pano_right.jpg') 

#convert to grayscale

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# sanity check for sufficient match points
#matches, locs1, locs2 = matchPics(img1, img2, opts)
#plotMatches(img1, img2, matches, locs1, locs2) #many matches identified, but takes long to compute

stitcher = cv2.createStitcher(try_use_gpu=False)  #true = use GPU to aid stitching
pano_output = stitcher.stitch((img1,img2)) ##SOME ERROR HERE
cv2.imshow("output", pano_output)
cv2.waitKey(0)
