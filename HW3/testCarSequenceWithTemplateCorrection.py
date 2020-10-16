import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation


parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=10000, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
print(seq.shape)
num_frames = seq.shape[2] #  if this bugs out, remember to copy from orig directory

rect = [59, 116, 145, 151]
x1 = rect[0]
y1 = rect[1]
x2 = rect[2]
y2 = rect[3]
width_rect = np.rint(x2-x1+1) 
height_rect = np.rint(y2-y1+1) 

rect2 = [59, 116, 145, 151]
x1_rect2 = rect2[0]
y1_rect2 = rect2[1]
x2_rect2 = rect2[2]
y2_rect2 = rect2[3]


#initialize parameters before iteration
i = 0 
rect_list = []
update_Template = True
T1 = seq[:,:,0]
sum_p = 0

for i in range(0,num_frames-1): # for OH: this i doesn't increase somehow
    if (update_Template==True):
        It = seq[:,:,i]

    print("frame num", i)
    
    It = seq[:,:,i]
    It1 = seq[:,:,i+1] #It+1 is the tracking rect in next image
    p = LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2))
    #template tracking
    sum_p += p.reshape(2,1)
    pstar = LucasKanade(T1, It1, rect2, threshold, num_iters, sum_p)
    
    epsilon = 1
    if(np.linalg.norm(sum_p-pstar)<epsilon):
        rect[0] = x1_rect2 + pstar[0,0]
        rect[1] = y1_rect2 + pstar[1,0]
        rect[2] = x2_rect2 + pstar[0,0]
        rect[3] = y2_rect2 + pstar[1,0]
    
    else:
        sum_p = sum_p = p
        update_Template = False # flag to stop update of tempalte in next step

    if i==1 or (i%100)==0:
        plt.figure()
        plt.imshow(seq[:,:,i],cmap='gray')
        box_rect = patches.Rectangle((int(rect[0]), int(rect[1])), width_rect, height_rect, fill=False, edgecolor='blue', linewidth=1)
        plt.gca().add_patch(box_rect)
        plt.title('frame %d'%i)
        #plt.show()

        #print out at frames 1, 100, 200, 300, 400

rect_copy = rect.copy() # make a copy of rects
rect_list.append(rect_copy)
np.save('../data/carseqrects-wrct.npy',rect_list)
print('Finished test car wrct')  

    