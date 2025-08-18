import numpy as np
import cv2 as cv

img = np.zeros((512,512,3),np.uint8)

cv.line(img,(0,0),(511,511),(0,0,255),4)

cv.rectangle(img,(250,50),(400,200),(0,255,0),2)

cv.circle(img,(100,300),80,(255,255,255),2)

font = cv.FONT_HERSHEY_SIMPLEX

cv.putText(img,'Osamah Saeed',(10,450),font,2,(255,255,255),5)

cv.imshow('Image',img)

cv.waitKey(0)
cv.destroyAllWindows()