import cv2 as cv
import numpy as np 

img = cv.imread('images/houghlines5.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,120)
lines = cv.HoughLinesP(edges,rho=1,
                    theta= np.pi/180.0,
                    threshold=20,
                    minLineLength=40,
                    maxLineGap=5)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv.imshow("E;dges",edges)
cv.imshow("Lines",img)
cv.waitKey(0)
cv.destroyAllWindows()