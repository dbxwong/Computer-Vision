
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

img1 = cv2.imread('D:/MRSD/0.4 Computer Vision (16-720AC)/HW2/data/pano_L.jpg')
img2 = cv2.imread('D:/MRSD/0.4 Computer Vision (16-720AC)/HW2/data/pano_R.jpg') 

#convert to grayscale, if required
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
img1 = img1.astype(np.uint8)
img2 = img2.astype(np.uint8)
# sanity check for sufficient match points
#matches, locs1, locs2 = matchPics(img1, img2, opts)
#plotMatches(img1, img2, matches, locs1, locs2) #many matches identified, but takes long to compute

stitcher = cv2.createStitcher(try_use_gpu=False)  #true = use GPU to aid stitching
pano_output = stitcher.stitch((img1,img2)) ##SOME ERROR HERE
cv2.imshow("output", pano_output)
cv2.waitKey(0)

'''
mport cv2
import numpy as np
img_ = cv2.imread('original_image_right.jpg')
#img_ = cv2.imread('original_image_left.jpg')
#img_ = cv2.resize(img_, (0,0), fx=1, fy=1)
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
img = cv2.imread('original_image_left.jpg')
#img = cv2.imread('original_image_right.jpg')
#img = cv2.resize(img, (0,0), fx=1, fy=1)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
# find key points
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
#cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_,kp1,None))
#FLANN_INDEX_KDTREE = 0
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks = 50)
#match = cv2.FlannBasedMatcher(index_params, search_params)
match = cv2.BFMatcher()
matches = match.knnMatch(des1,des2,k=2)
good = []
for m,n in matches:
    if m.distance < 0.03*n.distance:
        good.append(m)
draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=None,
                       flags=2)
img3 = cv2.drawMatches(img_,kp1,img,kp2,good,None,**draw_params)
#cv2.imshow("original_image_drawMatches.jpg", img3)
MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    #cv2.imshow("original_image_overlapping.jpg", img2)
else:
    print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))
dst = cv2.warpPerspective(img_,M,(img.shape[1] + img_.shape[1], img.shape[0]))
dst[0:img.shape[0],0:img.shape[1]] = img
cv2.imshow("original_image_stitched.jpg", dst)
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame
cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
#cv2.imsave("original_image_stitched_crop.jpg", trim(dst))
'''
