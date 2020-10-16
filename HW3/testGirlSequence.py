import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=0.05, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
#template_threshold = args.template_threshold ## from Piazza, not required
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

print(seq.shape) # for debugging
num_frames = seq.shape[2] #  if this bugs out, remember to copy from orig directory

x1 = rect[0]
y1 = rect[1]
x2 = rect[2]
y2 = rect[3]
width_rect = np.rint(x2-x1+1) # width of rect - width may not be int hence round up before processing
height_rect = np.rint(y2-y1+1) # height of rect - height may not be int hence round up before processing

#initialize parameters before iteration
i = 0 
rect_list = []

for i in range(0,num_frames-1): # for OH: this i doesn't increase somehow

    print("frame id", i)
    
    It = seq[:,:,i]
    It1 = seq[:,:,i+1] # from Piazza: It+1 is the tracking rect in next image
    p = LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2))
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]
    rect_copy = rect.copy() # make a copy of rects
    rect_list.append(rect_copy)
   
       
    if i==1 or i == 20 or i ==40 or i == 60 or i ==80 or (i%100)==0:
        plt.figure()
        plt.imshow(seq[:,:,i],cmap='gray')
        box_rect = patches.Rectangle((int(rect[0]), int(rect[1])), width_rect, height_rect, fill=False, edgecolor='red', linewidth=1)
        plt.gca().add_patch(box_rect)
        plt.title('frame %d'%i)
        plt.show()

    
np.save('../data/girlseqrects.npy',rect_list) 
print('Finished test girl')