import cv2
import numpy as np
import matplotlib.pyplot as plt

# تحميل صورة باللون الرمادي
img = cv2.imread(r"D:\IT\Level 4\Image Processing\Image-Processing-Course\Lab5\HW edge detection\images\test2.png", cv2.IMREAD_GRAYSCALE)

# نحسب التدرج باستخدام Sobel كطريقة تقريبية للـ Gradient
grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)   # ∂I/∂x
grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)   # ∂I/∂y

# حساب المقدار (magnitude) والزاوية (direction)
magnitude = cv2.magnitude(grad_x, grad_y)
direction = cv2.phase(grad_x, grad_y, angleInDegrees=True)

# عرض النتائج
plt.figure(figsize=(12,6))
plt.subplot(2,2,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(2,2,2), plt.imshow(grad_x, cmap='gray'), plt.title("Gradient X (∂I/∂x)")
plt.subplot(2,2,3), plt.imshow(grad_y, cmap='gray'), plt.title("Gradient Y (∂I/∂y)")
plt.subplot(2,2,4), plt.imshow(magnitude, cmap='gray'), plt.title("Gradient Magnitude")
plt.show()
