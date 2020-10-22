import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect (4x1 vector): Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Put your implementation here - reference: http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf 
    p = p0
    
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

    # in translation model jacobian is not related to coordinates
    jacobian = np.array([[1,0],[0,1]])
    dp = [[rows_img], [cols_img]] # to start the loop

    while np.square(dp).sum() > threshold: #TA said can ignore num_iters
                    
        # warp image using translation motion model
        x1_warp = x1+p[0]
        y1_warp = y1+p[1]
        x2_warp = x2+p[0]
        y2_warp = y2+p[1]
        
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

        #I is (nx2)
        I = np.vstack((Ix_warp.ravel(),Iy_warp.ravel())).T
        
        #compute delta 
        delta = np.matmul(I,jacobian)

        #compute H is (2,2)
        H = np.matmul(delta.T,delta)
        
        #compute dp
        dp = np.linalg.inv(H) @ (delta.T) @ error_Img
        
        if np.square(dp).sum() > threshold:
            print('threshold reached')
        
        #update p
        p[0] = p[0] + dp[0,0]
        p[1] = p[0] + dp[1,0]

    return p
    