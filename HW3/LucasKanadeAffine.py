import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
import cv2
from matplotlib import pyplot as plt

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
    """

    # put your implementation here - reference: http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf
    M = np.eye(3).astype(np.float32)
    
      
    dp = [[1], [0], [0], [0], [1], [0]]
    p = np.zeros(6)
    x1, y1, x2, y2 = 0, 0, It.shape[1], It.shape[0]
    rows, cols = It.shape

    # compute gradient and RBspline
    Iy, Ix = np.gradient(It1)
    y = np.arange(0, rows, 1)
    x = np.arange(0, cols, 1)     
    c = np.linspace(x1, x2, cols)
    r = np.linspace(y1, y2, rows)
    cc, rr = np.meshgrid(c, r)

    RBspline_It = RectBivariateSpline(y, x, It)
    T = RBspline_It.ev(rr, cc)
    RBspline_gx = RectBivariateSpline(y, x, Ix)
    RBspline_gy = RectBivariateSpline(y, x, Iy)
    RBspline_It1 = RectBivariateSpline(y, x, It1)

    while np.square(dp).sum() > threshold:

        W = np.array([[1.0 + p[0], p[1], p[2]],
                       [p[3], 1.0 + p[4], p[5]]])
    
        x1_warp = W[0,0] * x1 + W[0,1] * y1 + W[0,2]
        y1_warp = W[1,0] * x1 + W[1,1] * y1 + W[1,2]
        x2_warp = W[0,0] * x2 + W[0,1] * y2 + W[0,2]
        y2_warp = W[1,0] * x2 + W[1,1] * y2 + W[1,2]
    
        cols_warp = np.linspace(x1_warp, x2_warp, It.shape[1])
        rows_warp = np.linspace(y1_warp, y2_warp, It.shape[0])
        cols_mesh, rows_mesh = np.meshgrid(cols_warp, rows_warp)
        
        warpImg = RBspline_It1.ev(rows_mesh, cols_mesh)

        #compute error image
        #errImg is (n,1)
        error = T - warpImg
        errorImg = error.reshape(-1,1)
        
        #compute gradient
        Ix_warp = RBspline_gx.ev(rows_mesh, cols_mesh)
        Iy_warp = RBspline_gy.ev(rows_mesh, cols_mesh)
        #I is (n,2)
        I = np.vstack((Ix_warp.ravel(),Iy_warp.ravel())).T
        
        #evaluate delta = I @ jac is (nx6)
        delta = np.zeros((It.shape[0]*It.shape[1], 6))
   
        for i in range(It.shape[0]):
            for j in range(It.shape[1]):
                #I is (1x2) for each pixel
                #Jacobian is (2x6)for each pixel
                I_indiv = np.array([I[i*It.shape[1]+j]]).reshape(1,2)
                
                jac_indiv = np.array([[j, 0, i, 0, 1, 0],
                                      [0, j, 0, i, 0, 1]]) 
                delta[i*It.shape[1]+j] = np.matmul(I_indiv, jac_indiv)
        
        #compute Hessian Matrix
        #H is (6x6)
        H = np.matmul(delta.T, delta)
        
        #compute dp
        #dp is (6x6)@(6xn)@(nx1) = (6x1)
        dp = np.linalg.inv(H) @ (delta.T) @ errorImg
        
        #update parameters
        p[0] += dp[0,0]
        p[1] += dp[1,0]
        p[2] += dp[2,0]
        p[3] += dp[3,0]
        p[4] += dp[4,0]
        p[5] += dp[5,0]

    M =  np.array([[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]],[0,0,1]])   
    
    return M
    