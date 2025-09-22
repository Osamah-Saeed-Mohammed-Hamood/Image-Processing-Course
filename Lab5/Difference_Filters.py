import cv2 as cv
import matplotlib.pyplot as plt 
import numpy as np 

img = cv.imread('images/natural.jpg')
mask_vertical = np.array([
    [0,1,0],
    [0,1,0],
    [0,-1,0]
])
mask_horizontal = np.array([
    [0,0,0],
    [1,1,-1],
    [0,0,0]
])
mask_diagonal1 = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,-1]
])
mask_diagonal2 = np.array([
    [0,0,1],
    [0,1,0],
    [-1,0,0]
])

sharpened_img1 = cv.filter2D(src = img,ddepth = -1 ,kernel = mask_vertical)
sharpened_img1 = cv.filter2D(src = sharpened_img1,ddepth = -1 ,kernel = mask_horizontal)
sharpened_img1 = cv.filter2D(src = sharpened_img1,ddepth = -1 ,kernel = mask_diagonal1)
sharpened_img1 = cv.filter2D(src = sharpened_img1,ddepth = -1 ,kernel = mask_diagonal2)

img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
sharpened_img1 = cv.cvtColor(sharpened_img1,cv.COLOR_BGR2RGB)

fig ,axs = plt.subplots(1,2,figsize = (10,4))

axs[0].imshow(img)
axs[0].set_title('Original Image')

axs[1].imshow(sharpened_img1)
axs[1].set_title('Sharpened Image 1')

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()