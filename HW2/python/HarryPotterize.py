import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from planarH import computeH_ransac
import matplotlib.pyplot as plt
from matchPics import matchPics

#Import necessary functions
opts = get_opts()
# read cv_cover.jpg and hp_cover.jpg
# cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png') 
hp_cover =  cv2.imread('../data/hp_cover.jpg')


# compute homography automatically using MatchPics and computeH_ransac

matches, locs1, locs2 = matchPics(hp_cover, cv_desk, opts)
H = computeH_ransac(locs1, locs2, opts) #returns best H2to1 and inliers


# use cv2.warpPerspective function to wrap hp_cover.jpg to cv_desk.png image
warped_img = cv2.warpPerspective(hp_cover, H, cv_desk.shape[1],cv_desk.shape[0]) 

# print output
cv2.imshow("source image", hp_cover)
cv2.imshow("destination image", cv_desk)
cv2.imshow("warped image", warped_img)


# modify hp_cover.jpg to fix mispositioned image

# compute and output composite image
#compositeH(H2to1, template, img)

# Write script for Q2.2.4
