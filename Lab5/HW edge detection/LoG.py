import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"images\test2.png", cv2.IMREAD_GRAYSCALE)

log_kernel = np.array([[0, 0, -1, 0, 0],
                    [0, -1, -2, -1, 0],
                    [-1, -2, 16, -2, -1],
                    [0, -1, -2, -1, 0],
                    [0, 0, -1, 0, 0]], dtype=np.float32)

log_result = cv2.filter2D(img, -1, log_kernel)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(1,2,2), plt.imshow(log_result, cmap='gray'), plt.title("LoG (5x5)")
plt.show()
