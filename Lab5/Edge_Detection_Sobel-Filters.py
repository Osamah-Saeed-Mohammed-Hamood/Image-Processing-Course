import cv2 as cv

img = cv.imread('images/sunflower.jpg')
sobelx = cv.Sobel(img,ddepth=-1,dx=1,dy=0,ksize=5)
sobely = cv.Sobel(img,-1,0,1,5)
sobelxy = cv.Sobel(img,-1,1,1,5)

cv.imshow('Original Image',img)
cv.imshow('Sobelx Image',sobelx)
cv.imshow('Sobely Image',sobely)
cv.imshow('Sobelxy Image',sobelxy)

cv.waitKey(0)
cv.destroyAllWindows()
