import cv2 as cv
import numpy as np

img = cv.imread('images/Arithmetic.jpg')
img_float = np.float32(img) 

c = 255/np.log(1+np.max(img_float))

img_log = c*np.log(1+img_float)
img_log = np.uint8(img_log)

cv.imshow('Original Image', img)
cv.imshow('Log Transformation', img_log)

cv.waitKey(0)
cv.destroyAllWindows()