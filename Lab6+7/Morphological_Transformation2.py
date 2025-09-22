import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/i_noise.png',cv.IMREAD_GRAYSCALE)
kernel = np.ones((5,5),np.uint8)

opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)

plt.subplot(131),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)),plt.title('Original Image'),plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(cv.cvtColor(opening, cv.COLOR_BGR2RGB)),plt.title('Opening Image'),plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(cv.cvtColor(tophat, cv.COLOR_BGR2RGB)),plt.title('Tophat Image'),plt.xticks([]), plt.yticks([])
plt.tight_layout()  
plt.show()