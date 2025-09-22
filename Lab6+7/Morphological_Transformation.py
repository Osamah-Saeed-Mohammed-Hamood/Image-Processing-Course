import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/i.png',cv.IMREAD_GRAYSCALE)
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)
dilation = cv.dilate(img,kernel,iterations = 1)
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

plt.subplot(141),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)),plt.title('Original Image'),plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(cv.cvtColor(erosion, cv.COLOR_BGR2RGB)),plt.title('Erosion Image'),plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(cv.cvtColor(dilation, cv.COLOR_BGR2RGB)),plt.title('Dilation Image'),plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(cv.cvtColor(gradient, cv.COLOR_BGR2RGB)),plt.title('Gradient Image'),plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()