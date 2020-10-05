
import numpy as np
import cv2
from opts import get_opts
from planarH import compositeH

opts = get_opts()

#load images
img1 = cv2.imread('../data/pano_left.jpg')
img2 = cv2.imread('../data/pano_right.jpg') 
img_left = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img_right = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
# initiate ORB star detector and find keypoints
orb = cv2.ORB_create()
keypts1, des1 = orb.detectAndCompute(img_left,None)
keypts2, des2 = orb.detectAndCompute(img_right,None)

# Initiate SIFT detector - alternative
#sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT - alternative
#kp1, des1 = sift.detectAndCompute(img1,None)
#kp2, des2 = sift.detectAndCompute(img2,None)


# create BFMatcher object - reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# match descriptors
matches = bf.knnMatch(des1,des2,k=1)

good = []
for m,n in matches:
    if m.distance < 0.03*n.distance:
        good.append(m)

min_num_matches = 10
if len(good)>min_num_matches:
    matched_locs1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    matched_locs2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # compute homography
    H1to2, mask = cv2.findHomography(matched_locs1, matched_locs2, cv2.RANSAC, ransacReprojThreshold=2.0)
    
    # sitch composite impage
    composite_img = compositeH(H1to2, img1, img2)
    plt.imshow(composite_img)
    plt.show()

else:
    print('error: not enough matches between both images, please try another pair')