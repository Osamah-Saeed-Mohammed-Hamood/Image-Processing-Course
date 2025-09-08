import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"images\test2.png", cv2.IMREAD_GRAYSCALE)

prewitt_kernel_x = np.array([[-1,0,1],
                            [-1,0,1],
                            [-1,0,1]])

prewitt_kernel_y = np.array([[-1,-1,-1],
                            [0,0,0],
                            [1,1,1]])

prewitt_x = cv2.filter2D(img, -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(img, -1, prewitt_kernel_y)

prewitt_combined = cv2.magnitude(np.float32(prewitt_x), np.float32(prewitt_y))

plt.figure(figsize=(10,5))
plt.subplot(1,4,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(1,4,2), plt.imshow(prewitt_x, cmap='gray'), plt.title("Prewitt X")
plt.subplot(1,4,3), plt.imshow(prewitt_y, cmap='gray'), plt.title("Prewitt Y")
plt.subplot(1,4,4), plt.imshow(prewitt_combined, cmap='gray'), plt.title("Prewitt Combined")
plt.show()
