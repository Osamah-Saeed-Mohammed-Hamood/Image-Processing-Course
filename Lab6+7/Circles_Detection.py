import cv2 as cv
import numpy as np

planets = cv.imread('images/planet_glow.jpg')
gray_img = cv.cvtColor(planets,cv.COLOR_BGR2GRAY)
gray_img = cv.medianBlur(gray_img,5)
circles = cv.HoughCircles(gray_img,cv.HOUGH_GRADIENT,
                        1,120,param1=90,param2=40,
                        minRadius=0,maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv.circle(planets,(i[0],i[1]),i[2],(0,255,0),2)
        cv.circle(planets,(i[0],i[1]),2,(0,0,255),3)

cv.imwrite("images/planets_hough_circles.jpg",planets)
cv.imshow("Hough Circles",planets)
cv.waitKey(0)
cv.destroyAllWindows()