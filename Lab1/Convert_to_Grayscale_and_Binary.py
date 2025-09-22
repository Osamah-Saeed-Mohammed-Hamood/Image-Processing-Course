import cv2 as c
import numpy as np

img = c.imread("images/home.png")
grayimg = c.cvtColor(img,c.COLOR_BGR2GRAY)
ret,binaryimg = c.threshold(grayimg,127,255,c.THRESH_BINARY)

bwimg = np.zeros_like(grayimg)
bwimg[grayimg>140]=255

c.imshow("Color Image",img)
c.imshow("Gray Scale",grayimg)
c.imshow("Binary Image",binaryimg)
c.imshow("Binary Image 2",bwimg)

c.waitKey(0)
c.destroyAllWindows()