import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
import cv2
from scipy.ndimage.morphology import binary_dilation, binary_erosion

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)
    
    height, width = image2.shape

    # warp image It using M so that it's registered to It+1
    M = LucasKanadeAffine(image1,image2)
    image2_warp = cv2.warpAffine(image2, M, (width,height))
    image2_warp = binary_erosion(image2_wrap, iterations = 3) 
    image2_warp = binary_dilation(image2_warp, iterations = 3)
    #image2_warp = binary_dilation(image2_warp, iterations = 5)
    #image2_warp = binary_erosion(image2_wrap, iterations = 5)
    #image2_warp = binary_erosion(image2_wrap, iterations = 7)
    #image2_warp = binary_erosion(image2_wrap, iterations = 7)
    # subtract the above from It+1
    difference = np.absolute(image1-image2_warp)
    mask = mask > threshold
    
    return mask # n x m binary image of same size that dictates which pixels are considered corresponding to moving objects
