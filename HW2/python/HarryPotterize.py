import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from planarH import computeH_ransac
import matplotlib.pyplot as plt
from matchPics import matchPics


opts = get_opts()

# Read images
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png') 
hp_cover =  cv2.imread('../data/hp_cover.jpg')


# compute homography of cv_cover and cv_desk automatically using MatchPics
matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts)

#locs1[:[0,1]]=locs1[:[1,0]]
#locs2[:[0,1]]=locs2[:[1,0]]

bestH2to1, inliers = computeH_ransac(locs1, locs2, opts)

harryPotter_resized = cv2.resize(hp,cover, (cv_cover.shape[1], cv_cover.shape[0]))

compositeImg = compositeH(bestH2to1, harryPotter_resized, cv_desk)
plt.imshow(compositeImg)

compositeImg_RGB = cv2.cvtColor(compositeImg, cv2.GRAY2RGB) # or cv2.COLOR_BGR2RGB?
plt.imshow(compositeImg_RGB)

