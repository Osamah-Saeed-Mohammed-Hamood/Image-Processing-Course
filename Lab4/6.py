import cv2 as cv
import numpy as np

img = cv.imread('images/sunflower.jpg')
mask = np.ones((5,5), np.float32) / 25

img_filter = cv.filter2D(src = img,
                        ddepth=-1,
                        kernel=mask)
cv.imshow('Original Image', img)
cv.imshow('Filtered Image', img_filter)

cv.waitKey(0)
cv.destroyAllWindows()