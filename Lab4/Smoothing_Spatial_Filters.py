import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/sunflower.jpg')
img_meanfilter = cv.blur(img, (5,5))
img_Gaussianfilter = cv.GaussianBlur(img, (5,5), 0)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_meanfilter = cv.cvtColor(img_meanfilter, cv.COLOR_BGR2RGB)
img_Gaussianfilter = cv.cvtColor(img_Gaussianfilter, cv.COLOR_BGR2RGB)  

fig , axs = plt.subplots(1,3, figsize=(10,4))
axs[0].imshow(img)
axs[0].set_title('Original Image')
axs[1].imshow(img_meanfilter)
axs[1].set_title('Mean Filter Image')
axs[2].imshow(img_Gaussianfilter)
axs[2].set_title('Gaussian Filter Image')
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()