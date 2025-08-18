import cv2 as cv

img = cv.imread('images/planet_glow.jpg')
startrow = 155
endrow = 315
startcol = 440
endcol = 596

ROI = img[startrow:endrow,startcol:endcol]
ROI2 = img[152:295,211:446]

cv.imshow('Original image',img)
cv.imshow('Cropping',ROI)
cv.imshow('Cropping2',ROI2)

cv.waitKey()
cv.destroyAllWindows()