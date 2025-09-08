import cv2 as cv

img = cv.imread('images/number.PNG',0)
img_blur = cv.GaussianBlur(img,(3,3),0)
Laplacian = cv.Laplacian(img_blur,-1)
Canny = cv.Canny(img_blur,threshold1=100,threshold2=200)
cv.imshow('Original image',img)
cv.imshow('Laplacian',Laplacian)
cv.imshow('Canny',Canny)
cv.waitKey(0)
cv.destroyAllWindows()