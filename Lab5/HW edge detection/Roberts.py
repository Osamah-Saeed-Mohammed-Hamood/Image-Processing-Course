import cv2
import numpy as np
import matplotlib.pyplot as plt

# تحميل صورة باللون الرمادي
img = cv2.imread(r"D:\IT\Level 4\Image Processing\Image-Processing-Course\Lab5\HW edge detection\images\test2.png", cv2.IMREAD_GRAYSCALE)

# أقنعة Roberts (2x2)
roberts_cross_x = np.array([[1, 0],
                            [0, -1]], dtype=np.float32)

roberts_cross_y = np.array([[0, 1],
                            [-1, 0]], dtype=np.float32)

# تطبيق الفلترة
roberts_x = cv2.filter2D(img, -1, roberts_cross_x)
roberts_y = cv2.filter2D(img, -1, roberts_cross_y)

# دمج الاتجاهين
roberts_combined = cv2.magnitude(np.float32(roberts_x), np.float32(roberts_y))

# عرض النتائج
plt.figure(figsize=(10,5))
plt.subplot(1,4,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(1,4,2), plt.imshow(roberts_x, cmap='gray'), plt.title("Roberts X")
plt.subplot(1,4,3), plt.imshow(roberts_y, cmap='gray'), plt.title("Roberts Y")
plt.subplot(1,4,4), plt.imshow(roberts_combined, cmap='gray'), plt.title("Roberts Combined")
plt.show()
