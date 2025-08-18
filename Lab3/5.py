import cv2 as cv 
import numpy as np 
from matplotlib import pyplot as plt 

img1 = cv.imread('images/Arithmetic.jpg')
img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)

res = cv.multiply(img1,2)

res_divided = cv.divide(img1,2)

fig , axs = plt.subplots(1,3,figsize = (10,4))

axs[0].imshow(img1)
axs[0].set_title('Image1')
axs[1].imshow(res)
axs[1].set_title('Multiply Image by 2')
axs[2].imshow(res_divided)
axs[2].set_title('Divide Image by 2')

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()