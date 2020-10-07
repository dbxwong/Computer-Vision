import numpy as np
import cv2

## COLLABORATORS: CORINNE ALINI, HUSAM WADI, DANIEL BRONSTEIN, LIU JINKUN, JONATHAN SCHWARTZ, AARUSHI WADHWA

def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points

	# x1 x2 are coord of 2 different images; x1 = img 1, x2 = img2;
	# x1 and x2 are N X 2 matrices

	n = x1.shape[0]
	if x2.shape[0] !=n:
		print('Error: number of points dont match')
	
	N = x1.shape[0]
	
	A = np.zeros((8, 9))

# Compute homography matrix
	for i in range(N):
		
		u,v = x1[i][0], x1[i][1]
		x,y = x2[i][0], x2[i][1]
		
		A[2*i]=[x,y,1,0,0,0,-u*x, -u*y, -u] #append matrix A
		A[2*i+1]=[0,0,0,x,y,1,-v*x, -v*y, -v]

	A = np.asarray(A)
	#invA = inv(A)
	#V = eig(inv(A)*A)
	#U,S,V = np.linalg.svd(A) # V columns corresppond to the eigenvectors of A^-1A or use SVD
	#H2to1 = np.reshape(V[8,:],(3,3))

	V = np.matmul(np.transpose(A),A)
	
	eig_vals, eig_vecs = np.linalg.eig(V)

	H = eig_vecs[:,np.argmin(eig_vals)]
	H2to1=np.reshape(H,(3,3))


	return H2to1


def computeH_norm(x1, x2): # this function normalizes the cooredinates
	#Q2.2.2
	# x1 and x2 are N X 2 matrices containing point pairs between two images
	
	n = x1.shape[0]
	if x2.shape[0] !=n:
		print('Error: number of points dont match')

	#Shift the origin of the points to the centroid
	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	## get mean of points (centroid) in x1 and x2
	x1_arr = np.asarray(x1)
	x2_arr = np.asarray(x2)

	#flattened_coords = np.hstack((x1_arr, x2_arr))
	flattened_coords = np.flatten((x2_arr,x2_arr))

	mean_x1, mean_y1, mean_x2, mean_y2 = np.mean(flattened_coords, axis=0)
	 	# shift origin of points to centroid (points - mean)
	shiftedPoints = flattened_coords - [mean_x1, mean_y1, mean_x2, mean_y2] 
	
	
	# compute largest scale factor
	sx1, sy1, sx2, sy2 = np.max(np.abs(shiftedPoints),axis=0)

	T1 = np.array([[1/sx1, 0, -x_mean1/sx1], 
					[0, 1/sy1, -y_mean1/sy1], 
					[0, 0, 1]])  # this T matrix is to reverse the normalization

	T2 = np.array([[1/sx2, 0, -x_mean2/sx2], 
					[0, 1/sy2, -y_mean2/sy2], 
					[0, 0, 1]])  # this T matrix is to reverse the normalization; T is a 3x3 matrix

	#compute normalized homogenous coordinate x1_homo = T1*x1;
	 
	x1_hom = np.hstack((x1_arr, np.ones((x1_arr.shape[0],1))))
	x2_hom = np.hstack((x2_arr, np.ones((x2_arr.shape[0],1))))
	
	#normalize
	for row in range(x1_hom.shape[0]):
		x1_norm = np.asarray([np.matmul(T1, x1_hom[row])[:2]])

	for row in range(x2_hom.shape[0]):
		x2_norm = np.asarray([np.matmul(T2, x2_hom[row])[:2]])

	#Compute homography
	H = computeH(x1_norm, x2_norm)

	#Denormalization H = inv(T2)*H*T1
	T2inv = np.linalg.inv(T2)
	H2to1 = np.matmul(T2inv,np.matmul(H, T1))

	return H2to1



def computeH_ransac(locs1, locs2, opts):
		
	'''
	Compute the best fitting homography given a list of matching points
	input: locs1, locs2 - N x 2 matrices of matched points
	
	use computeH_norm to compute the homography

	output: bestH2to1 - homography H with most inliers found during RANSAC. x2 is a point in locs2; x1 is a corresponding point in locs1. 
			inliers - vector of length N with a 1 at those matches that are part of consensus set, else 0
	'''
	
	#Q2.2.3	
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for = default 500
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier = default 2
	x2_homo=[]

	for iters in range (max_iters):
		idx = np.random.choice(range(locs2.shape[0]),4)

		for i in range(idx):
			sampled_locs1 = locs1[i] # sampled 4 points from locs1
			sampled_locs2 = locs2[i] # sampled 4 points from locs1
			sampled_H = computeH_norm(sampled_locs1,sampled_locs2) #compute homography of 4 sampled points 

			#check homography of sampled points against x2
			x2_homo = np.append(np.transpose(locs2), np.ones((locs2.shape[0],1)), axis=0)

			homography_check = np.matmul(sampled_H,x2_homo) 

			distance = np.linalg.norm(locs1 - homography_check,axis=0) #axis=0 or 1
			squared_dist = distance[0,i]**2 + distance[1,i]**2
			print(squared_dist)

			# count inliers based on inlier tolerance
			inliers_count = 0
			max_inlier_count = 0

			if squared_dist < inlier_tol**2:
				inlier_count +=1

			if inliers_count > max_inlier_count:
				bestH2to1 = sampled_H # save sampledH as best H
				max_inlier_count = inliers_count # save new inlier count as max inlier count
				inliers = max_inlier_count # to return inliers as HW question requires 

	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography 	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.

	# Create mask of same size as template
	template_warp = cv2.warpPerspective(template,H2to1,(img.shape[1],img.shape[0]))

	# Initialize blank template and warp it with H2to1
	template_blank = np.ones(template.shape[:2], dtype = np.uint8) 
	template_blank_warped = cv2.warpPerspective(template_blank, H2to1, (img.shape[1],img.shape[0]))

	# Invert blank template, apply template to mask img
	template_blank_inversed = cv2.bitwise_not(template_blank_warped)
	masked_img = cv2.bitwise_and(img, img, mask=template_blank_inversed)

	#Use mask to combine the warped template and the image
	composite_img = cv2.bitwise_or(masked_img, template_warp)

	return composite_img


