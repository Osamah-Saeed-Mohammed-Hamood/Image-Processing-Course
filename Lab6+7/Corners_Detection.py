import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/corners.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray, maxCorners=0,
                                qualityLevel=0.01, minDistance=10)

corners = np.intp(corners)
for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 4, (255,0,0), -1)

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Corners Detection')
plt.show()