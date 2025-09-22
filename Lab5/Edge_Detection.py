import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np 

img = cv.imread('images/coins.jpg')
mask1 = np.array([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]
])
mask2 = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])
mask3 = np.array([
    [-2,1,-2],
    [1,4,1],
    [-2,1,-2]
])

sharpened_img1 = cv.filter2D(src = img,ddepth = -1,kernel = mask1)
sharpened_img2 = cv.filter2D(src = img,ddepth = -1,kernel = mask2)
sharpened_img3 = cv.filter2D(src = img,ddepth = -1,kernel = mask3)

img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
sharpened_img1 = cv.cvtColor(sharpened_img1,cv.COLOR_BGR2RGB)
sharpened_img2 = cv.cvtColor(sharpened_img2,cv.COLOR_BGR2RGB)
sharpened_img3 = cv.cvtColor(sharpened_img3,cv.COLOR_BGR2RGB)

fig , axs = plt.subplots(1,4,figsize = (10,4))

axs[0].imshow(img)
axs[0].set_title('Original Image')
axs[1].imshow(sharpened_img1)
axs[1].set_title('Sharpened Image 1')
axs[2].imshow(sharpened_img2)
axs[2].set_title('Sharpened Image 2')
axs[3].imshow(sharpened_img3)
axs[3].set_title('Sharpened Image 3')

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()