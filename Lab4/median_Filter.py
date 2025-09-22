import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('images/noisysalterpepper.png')
img_medianfilter = cv.medianBlur(img,5)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_medianfilter = cv.cvtColor(img_medianfilter, cv.COLOR_BGR2RGB)  

fig , axs = plt.subplots(1,2, figsize=(10,4))
axs[0].imshow(img)
axs[0].set_title('Original Image')
axs[1].imshow(img_medianfilter)
axs[1].set_title('Median Filtered Image')
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()