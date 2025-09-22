import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

img1 = cv.imread('images/balloons.jpg')
img2 = cv.imread('images/boat.jpg')
img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(img2,cv.COLOR_BGR2RGB)

img2 = cv.resize(img2,(400,400))
img1 = cv.resize(img1,(400,400))

res = cv.addWeighted(img1,0.7,img2,0.3,0)
fig , axs = plt.subplots(1,3,figsize = (10,4))

axs[0].imshow(img1)
axs[0].set_title('Image1')
axs[1].imshow(img2)
axs[1].set_title('Image2')
axs[2].imshow(res)
axs[2].set_title('Add Image')

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()