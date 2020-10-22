import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion


# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1000, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=5, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask') #defauylt 0.2
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')
num_frames = seq.shape[2]


masks=[]
for i in range(0, num_frames-1):

    print('frame: ', i)
    mask = SubtractDominantMotion(seq[:,:,i], seq[:,:,i+1], threshold,num_iters,tolerance)
    img1 = seq[:,:,i-1]
    img2 = seq[:,:,i]
    #masks = mask[:,:,i-1]
    
    highlight = np.where(mask,1.0,img1)
    img_stack = np.dstack((img1,img1))
    superimpose_img = np.dstack((img_stack,highlight))
    
    plt.imshow(superimpose_img)
    #plt.show()
    
    if i in [30, 60, 90, 120]:
        plt.savefig("antseq"+str(i)+".png")
    plt.pause(1.0)
    