import cv2

img = cv2.imread("images/Qassam.png")
cv2.imshow("image 1",img)
cv2.waitKey(0)
cv2.imwrite("home.png",img)
cv2.destroyAllWindows()