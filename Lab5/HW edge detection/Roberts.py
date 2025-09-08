import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"images\test2.png", cv2.IMREAD_GRAYSCALE)

roberts_cross_x = np.array([[1, 0],
                            [0, -1]], dtype=np.float32)

roberts_cross_y = np.array([[0, 1],
                            [-1, 0]], dtype=np.float32)

roberts_x = cv2.filter2D(img, -1, roberts_cross_x)
roberts_y = cv2.filter2D(img, -1, roberts_cross_y)

roberts_combined = cv2.magnitude(np.float32(roberts_x), np.float32(roberts_y))

plt.figure(figsize=(10,5))
plt.subplot(1,4,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(1,4,2), plt.imshow(roberts_x, cmap='gray'), plt.title("Roberts X")
plt.subplot(1,4,3), plt.imshow(roberts_y, cmap='gray'), plt.title("Roberts Y")
plt.subplot(1,4,4), plt.imshow(roberts_combined, cmap='gray'), plt.title("Roberts Combined")
plt.show()
