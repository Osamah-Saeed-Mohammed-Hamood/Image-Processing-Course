import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/water_coins.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret , thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

kernal = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernal,iterations=2)
sure_bg = cv.dilate(opening,kernal,iterations=3)
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

ret, markers = cv.connectedComponents(sure_fg)

markers = markers+1

markers[unknown==255] = 0
markers = cv.watershed(img,markers)
seg = img.copy()
seg[markers == -1] = [255,0,0]

plt.figure(figsize=(12, 12))

plt.subplot(3,3,1),plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB)),plt.title('Original Image'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,2),plt.imshow(cv.cvtColor(thresh,cv.COLOR_BGR2RGB)),plt.title('Thresh Image'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,3),plt.imshow(cv.cvtColor(opening,cv.COLOR_BGR2RGB)),plt.title('Opening Image'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,4),plt.imshow(cv.cvtColor(sure_bg,cv.COLOR_BGR2RGB)),plt.title('Sure_bg Image'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,5),plt.imshow(cv.cvtColor(dist_transform,cv.COLOR_BGR2RGB)),plt.title('Distance Transform Image'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,6),plt.imshow(cv.cvtColor(sure_fg,cv.COLOR_BGR2RGB)),plt.title('Sure Foreground Area Image'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,7),plt.imshow(cv.cvtColor(unknown,cv.COLOR_BGR2RGB)),plt.title('Subtract Image'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,8),plt.imshow(markers,cmap="tab20b"),plt.title('Markers Image'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,9),plt.imshow(cv.cvtColor(seg, cv.COLOR_BGR2RGB)),plt.title('Segmented Image'),plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()