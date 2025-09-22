import cv2 as cv
import numpy as np

img_noisy1 = cv.imread('images/noisysalterpepper.png', 0)
m , n = img_noisy1.shape

img_new1 = np.zeros([m,n])
for i in range(1,m-1):
    for j in range(1,n-1):
        temp = [img_noisy1[i-1,j-1],
            img_noisy1[i-1,j],
            img_noisy1[i-1,j+1],
            img_noisy1[i,j-1],
            img_noisy1[i,j], 
            img_noisy1[i,j+1],
            img_noisy1[i+1,j-1],
            img_noisy1[i+1,j],
            img_noisy1[i+1,j+1]]
        
        temp = sorted(temp)
        img_new1[i,j] = temp[4]

img_new1 = img_new1.astype(np.uint8)
cv.imwrite('images/new_medianfilter_saltandpepper.png', img_new1)
cv.imshow('Median Filtered Image', img_new1)

cv.waitKey(0)
cv.destroyAllWindows()