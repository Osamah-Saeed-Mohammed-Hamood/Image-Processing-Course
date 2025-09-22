import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/i_noisee.png',cv.IMREAD_GRAYSCALE)
kernel = np.ones((5,5),np.uint8)

closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

plt.subplot(131),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)),plt.title('Original Image'),plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(cv.cvtColor(closing, cv.COLOR_BGR2RGB)),plt.title('Closing Image'),plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(cv.cvtColor(blackhat, cv.COLOR_BGR2RGB)),plt.title('Blackhat Image'),plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()