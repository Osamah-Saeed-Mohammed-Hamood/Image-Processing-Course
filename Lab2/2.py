import cv2 as cv

image = cv.imread('images/planet_glow.jpg')
scale_factor_1 = 2.0
scale_factor_2 = 1/2.0
height , width = image.shape[:2]
new_height = int(height * scale_factor_1)
new_width = int (width * scale_factor_1)

zoomed_image = cv.resize(src = image,dsize=(new_width,new_height),interpolation = cv.INTER_CUBIC)

new_height1 = int(height*scale_factor_2)
new_width1 = int(width*scale_factor_2)

Shrink_image = cv.resize(src=image , dsize=(new_width1,new_height1),interpolation=cv.INTER_AREA)

cv.imshow('Image',image)
cv.imshow('Zoomed Image',zoomed_image)
cv.imshow('Shrink Image',Shrink_image)
cv.waitKey()
cv.destroyAllWindows()