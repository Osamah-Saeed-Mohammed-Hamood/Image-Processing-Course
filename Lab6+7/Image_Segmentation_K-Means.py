import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/images1.webp')
Z = img.reshape((-1,3))
Z = np.float32(Z)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
_, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
segmented_image = res.reshape((img.shape))

plt.imshow(cv.cvtColor(segmented_image, cv.COLOR_BGR2RGB))
plt.title('Segmented Image with K-Means Clustering')
plt.show()
