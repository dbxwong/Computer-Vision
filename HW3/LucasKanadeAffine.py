import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.eye(3)
    p = np.zeros(6)
 
    x1, y1= rect[0], rect[1]
    x2, y2 = rect[2], rect[3]
    
    rows_img, cols_img = It.shape
    rows_rect = np.rint(x2 - x1)
    cols_rect = np.rint(y2-y1)
    rows_rect, cols_rect = int(rows_rect), int(cols_rect) #had trouble with .0s so enforce as int
    rows_img, cols_img = int(rows_img), int(cols_img) #had trouble with .0s so enforce as int
 
    # compute mesh 
    Iy, Ix = np.gradient(It1)
    y = np.arange(0, rows_img, 1)
    x = np.arange(0, cols_img, 1)     
    x_inter = np.linspace(x1, x2, cols_rect)
    y_inter = np.linspace(y1, y2, rows_rect)
    mesh_cols, mesh_rows = np.meshgrid(x_inter, y_inter)

    #It_spline=RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It) - guide from Piazza
    RectBVspline_It = RectBivariateSpline(y, x, It)
    RectBVspline_Ix = RectBivariateSpline(y, x, Ix)
    RectBVspline_Iy = RectBivariateSpline(y, x, Iy)
    RectBVspline_It1 = RectBivariateSpline(y, x, It1)

    T = RectBVspline_It.ev(mesh_rows, mesh_cols)

    dp = [[rows_img], [cols_img]] # to start the loop
    M = []

    while np.square(dp).sum() > threshold: #TA said can ignore num_iters

        # affine warp - from lecture notes slide 27            
        W = np.array([1.0+p[0],p[1],p[2]], 
                    [p[3], 1+p[4]], p[5])
                            
        # warp image 
        x1_warp = W[0,0] * x1 + W[0,1] * y1 + W[0,2]
        y1_warp = W[1,0] * x1 + W[1,1] * y1 + W[1,2]
        x2_warp = W[0,0] * x1 + W[0,1] * y1 + W[0,2]
        y2_warp = W[1,0] * x1 + W[1,1] * y1 + W[1,2]
        
        cols_warp = np.linspace(x1_warp, x2_warp, cols_rect)
        rows_warp = np.linspace(y1_warp, y2_warp, rows_rect)
        Xmesh, Ymesh = np.meshgrid(cols_warp, rows_warp)
        
        warpImg = RectBVspline_It1.ev(Ymesh, Xmesh)
        
        #error image
        error = T - warpImg
        error_Img = error.reshape(-1,1) 
        
        #gradient
        Ix_warp = RectBVspline_Ix.ev(Ymesh, Xmesh)
        Iy_warp = RectBVspline_Iy.ev(Ymesh, Xmesh)

        #I is nx2
        I = np.vstack((Ix_warp.ravel(),Iy_warp.ravel())).T
        
        #initiatlize delta - to compute delta np.matmul(I,jacobian) delta n x 6
        delta = np.zeros((It.shape[0]*It.shape[1],6))

        # compute I and Jacobian for pixels in region common to It
        for i in range(It.shape[0]):
            for j in range(It.shape[1]):
                pixel_jacobian = np.array([[j, 0, i, 0, 1, 0], # see lecture notes slide 27. jacobian is 2 x 6 for each pixel
                                    [0, j, 0, i, 0, 1]])

                pixel_I = np.array([I[i*It.shape[1]+j]]).reshape(1,2) # I is 1 x 2 for each pixel
                
                #compute delta
                delta[i*It.shape[1]+j] = np.matmul(pixel_I,pixel_jac)


        #compute H is 6 x 6
        H = np.matmul(delta.T,delta)
        
        #compute dp
        dp = np.linalg.inv(H) @ (delta.T) @ error_Img #dp is 6 x 1

        if np.square(dp).sum() > threshold:
            print('threshold reached')
        
        #update parmeters
        p[0] = p[0] + dp[0,0]
        p[1] = p[1] + dp[1,0]
        p[2] = p[2] + dp[2,0]
        p[3] = p[3] + dp[3,0]
        p[4] = p[4] + dp[4,0]
        p[5] = p[5] + dp[5,0]
     
    #update M (3x3)
    #M[0][0] = 1.0 + p[0] - some unresolvable indent issue
	#M[0][1] = p[1]
	#M[0][2] = p[2]
	#M[1][0] = p[3]
	#M[1][1] = 1.0 + p[4]
	#M[1][2] = p[5]
    M =  np.array([[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]]])   # M is transformation matrix relating image pair It and It+1 
    print('M=',M)
    print('p=',p)
    
    return M
    