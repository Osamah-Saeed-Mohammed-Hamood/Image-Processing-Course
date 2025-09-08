import cv2
import numpy as np
import matplotlib.pyplot as plt

# تحميل صورة باللون الرمادي
img = cv2.imread(r"D:\IT\Level 4\Image Processing\Image-Processing-Course\Lab5\HW edge detection\images\test2.png", cv2.IMREAD_GRAYSCALE)

# قناع LoG 5x5
log_kernel = np.array([[0, 0, -1, 0, 0],
                    [0, -1, -2, -1, 0],
                    [-1, -2, 16, -2, -1],
                    [0, -1, -2, -1, 0],
                    [0, 0, -1, 0, 0]], dtype=np.float32)

# تطبيق الفلترة
log_result = cv2.filter2D(img, -1, log_kernel)

# عرض النتائج
plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(1,2,2), plt.imshow(log_result, cmap='gray'), plt.title("LoG (5x5)")
plt.show()
