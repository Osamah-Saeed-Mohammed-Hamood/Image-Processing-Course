import cv2 as c
import sys

img = c.imread("images/image.jpg",0)
if img is None:
    print ("Filled to read image from file")
    sys.exit(1)
c.imshow("Image",img)

k = c.waitKey(0)
if k == 27:
    c.destroyAllWindows()
elif k == ord('s'):
    c.imwrite("Image2.png",img)
    c.destroyAllWindows()
