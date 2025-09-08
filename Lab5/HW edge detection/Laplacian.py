import cv2
import numpy as np
import matplotlib.pyplot as plt

# تحميل صورة باللون الرمادي
img = cv2.imread(r"D:\IT\Level 4\Image Processing\Image-Processing-Course\Lab5\HW edge detection\images\test2.png", cv2.IMREAD_GRAYSCALE)

# تطبيق مرشح Laplacian الجاهز من OpenCV
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)

# أمثلة لأقنعة Laplacian (3x3) لو حابب تطبقها يدوي
lap1 = np.array([[0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]], dtype=np.float32)

lap2 = np.array([[-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]], dtype=np.float32)

lap3 = np.array([[0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]], dtype=np.float32)

# تطبيق الفلترة اليدوية باستخدام الأقنعة
lap1_result = cv2.filter2D(img, -1, lap1)
lap2_result = cv2.filter2D(img, -1, lap2)
lap3_result = cv2.filter2D(img, -1, lap3)

# عرض النتائج
plt.figure(figsize=(12,8))
plt.subplot(2,3,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(2,3,2), plt.imshow(laplacian, cmap='gray'), plt.title("Laplacian (OpenCV)")
plt.subplot(2,3,3), plt.imshow(lap1_result, cmap='gray'), plt.title("Mask 1")
plt.subplot(2,3,4), plt.imshow(lap2_result, cmap='gray'), plt.title("Mask 2")
plt.subplot(2,3,5), plt.imshow(lap3_result, cmap='gray'), plt.title("Mask 3")
plt.show()
