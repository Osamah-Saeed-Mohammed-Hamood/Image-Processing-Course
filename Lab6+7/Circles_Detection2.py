import cv2 as cv
import numpy as np
img = cv.imread('images/foot_ball.png')
output = img.copy()
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray,5)

circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,
                        dp=1.2,minDist=50,
                        param1=100,param2=30,
                        minRadius=20,maxRadius=80)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x,y,r) in circles[0,:]:
        mask = np.zeros_like(gray)
        cv.circle(mask,(x,y),r,255,-1)

        y1,y2 = y-r,y+r
        x1,x2 = x-r,x+r
        ball_crop = img[y1:y2,x1:x2]
        mask_crop = mask[y1:y2,x1:x2]

        ny , nx = 78,58
        roi = output[ny:ny+ball_crop.shape[0],nx:nx+ball_crop.shape[1]]
        ball_area = cv.bitwise_and(ball_crop,ball_crop,mask=mask_crop)
        bg_area = cv.bitwise_and(roi,roi,mask=cv.bitwise_not(mask_crop))
        combined = cv.add(bg_area,ball_area)

        output[ny:ny+ball_crop.shape[0],nx:nx+ball_crop.shape[1]] = combined

cv.imshow("Original Image",img)
cv.imshow("Ball Copied",output)
cv.waitKey(0)
cv.destroyAllWindows()