import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"images\test2.png", cv2.IMREAD_GRAYSCALE)

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # محور X
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # محور Y

sobel_combined = cv2.magnitude(sobel_x, sobel_y)

plt.figure(figsize=(10,5))
plt.subplot(1,4,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(1,4,2), plt.imshow(sobel_x, cmap='gray'), plt.title("Sobel X")
plt.subplot(1,4,3), plt.imshow(sobel_y, cmap='gray'), plt.title("Sobel Y")
plt.subplot(1,4,4), plt.imshow(sobel_combined, cmap='gray'), plt.title("Sobel Combined")
plt.show()
