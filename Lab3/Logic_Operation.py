import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

img1 = cv.imread('images/paper.jpeg')
img2gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
ret , mask = cv.threshold(img2gray,190,255,cv.THRESH_BINARY)
img_inv = cv.bitwise_not(img1)
img2_fg = cv.bitwise_or(img1,img1,mask = mask)
img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
img2_fg = cv.cvtColor(img2_fg,cv.COLOR_BGR2RGB)
mask = cv.cvtColor(mask,cv.COLOR_BGR2RGB)
img_inv = cv.cvtColor(img_inv,cv.COLOR_BGR2RGB)

fig , axs = plt.subplots(1,4,figsize = (10,4))

axs[0].imshow(img1)
axs[0].set_title('Image1')
axs[1].imshow(mask)
axs[1].set_title('Mask - Binary Image')
axs[2].imshow(img2_fg)
axs[2].set_title('or Image')
axs[3].imshow(img_inv)
axs[3].set_title('not Image')

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()