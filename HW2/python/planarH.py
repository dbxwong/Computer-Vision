import numpy as np
import cv2


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points

	# x1 x2 are coord of 2 different images; x1 = img 1, x2 = img2;
	# x1 and x2 are N X 2 matrices


	n = x1.shape[0]
	if x2.shape[0] !=n:
		print('Error: number of points dont match')
	
	N = x1.shape[0]
	
	A = np.zeros(2*N, 9)

# Compute homography matrix
	for i in range(N):

		x,y = x1[i][0], x1[i][1]
		u,v = x2[i][0], x2[i][1]

		A[2*i]=[x,y,1,0,0,0,-u*x, -u*y, -u] #append matrix A
		A[2*i+1]=[0,0,0,x,y,1,-v*x, -v*y, -v]

	A = np.asarray(A)
	U,S,V = np.linalg.svd(A) # V columns corresppond to the eigenvectors of A^-1A
	h = V[:,end] # From the SVD we extract the right singular vector from V which corresponds to the smallest singular value
	H2to1 = h.reshape(3,3) # reshape H into a 3x3 matrix
     
	return H2to1


def computeH_norm(x1, x2): # this function normalizes the cooredinates
	#Q2.2.2
	# x1 and x2 are N X 2 matrices containing point pairs between two images

	x1_arr = np.asarray(x1)
	x2_arr = np.asaray(x2)

	n = x1.shape[0]
	if x2.shape[0] !=n:
		print('Error: number of points dont match')

	#Compute the centroid of the points 
	#Shift the origin of the points to the centroid
	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	x1 = x1 / x1[0]	
	mean_x1 = np.mean(x1[:1], axis=1)
	S1 = np.sqrt(2)/np.max(x1[0:])

	T1 = np.array([[S1 , 0 , -S1*mean_x1[0]], [ 0 , S1, -S1*mean_x1[1]] , [0,0,1]]) #T = [[1/sx, 0, -mx/sx], [0, 1/sy, -my/sy], [0, 0, 1]] normalizing transformation in matrix form
	x1 = np.dot(T1,x1)

	x2 = x2 / x2[0]	
	mean_x2 = np.mean(x2[:1], axis=1)
	S2 = np.sqrt(2)/np.max(x2[0:]) # use max value not std dev

	T2 = np.array([[S2 , 0 , -S2*mean_x2[0]], [ 0 , S2, -S2*mean_x2[1]] , [0,0,1]]) #T = [[1/sx, 0, -mx/sx], [0, 1/sy, -my/sy], [0, 0, 1]] normalizing transformation in matrix form
	x2 = np.dot(T2,x2)

	#compute homogenous coordinates 
	x1_hom = np.hstack((x2, np.ones((x1_array.shape[0],1))))
	x2_hom = np.hstack((x2, np.ones((x2_array.shape[0],1))))

	for row in range(x1_hom.shape[0]):
		x1_norm = np.asarray([np.matmlu(T1, x1_hom[row]), (0:2)] )## SOME BUG HERE at [0:2 or (0:2)?

	for row in range (x2_hom.shape[0]):
		x2_norm = np.asarray([np.matmlu(T2, x2_hom[row]), (0:2)]) ## SOME BUG HERE

	#Compute homography
	normed_H = HcomputeH(x1_norm, x2_norm)

	#Denormalization H = inv(T2)*H*T1
	T2inv = np.linalg.inv(T2)
	H2to1 = T2inv * normed_H * T1

	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	
	'''
	Compute the best fitting homography given a list of matching points
	input: locs1, locs2 - N x 2 matrices of matched points
	
	## use computeH_norm to compute the homography

	output: bestH2to1 - homography H with most inliers found during RANSAC. x2 is a point in locs2; x1 is a corresponding point in locs1. 
			inliers - vector of length N with a 1 at those matches that are part of consensus set, else 0

	'''
	
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for = default 500
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier = default 2
	
	if locs2.shape[0] !=n:
		print('Error: number of points dont match')
		
	bestH2to1=[]
	inliers=[]
	n = locs1.shape[0]

	for i in range(max-iters):
		idx = np.random.choice (n,4, replace=False) # random selection of data set (4 points) which contains outliers
		locs1_samp = np.array(locs1[idx] for idx in idx)
		locs1_samp = np.array(locs2[idx] for idx in idx)
		h = computeH_norm(locs1_samp, locs2_samp)


		# Comput Inliers - compute reprojection error and squared error distance
		X_homo = np.append(np.transpose(locs1),np.ones(n,1), axis = 0) # N x 2 transposed to 2 x N
		U_homo = np.append(np.transpose(locs2),np.ones(n,1), axis = 0) 
		reproj = np.matmul(h, U_homo) 
		reproj_norm = np.divide(reproj, reproj[2,:])
			
		maxNum_inliers= 0 #initialize

		error = X_homo-reproj_norm
			
		for i in range(n):
			sq_dist = error[0,i]**2 + error[1,i] # compute squared error distance before comparing with tolerance^2
			if sq_dist <= (inlier_tol**2):
				inliers+=1
				
			print(inliers)

			if inliers > maxNum_inliers:
				bestH2to1 = h #assign new H to best H
				inliers = maxNum_inliers #assign new inlier number to maxNumber of inliers
			
			print("max number of RANSAC inliers: ", maxNum_inliers)
	
	## early termination - to implement if there's time:
	# when inlier ratio reaches expected ratio of inliers
	# T = (1-e)*(total number of data points)

	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography
	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.

	# Create mask of same size as template
	template_warp = cv2.warpPerspective(template,H2to1,(img.shape[1],img.shape[0]))

	# Initialize blank template (all black) and warp with H2to1
	template_blank = 0*np.ones(template.shape[:2], dtype = np.uint8)
	template_blank_warped = cv2.warpPerspective(template_blank, H2to1, (img.shape[1],img.shape[0]))

	# Invert blank template, apply template to mask img
	template_blank_inversed = cv.bitwise_not(template_blank_warped)
	masked_img = cv2.bitwise_and(img, img, mask=template_blank_inversed)

	#Use mask to combine the warped template and the image
	composite_img = cv2.bitwise_or(masked_img, template_warp)

	return composite_img


