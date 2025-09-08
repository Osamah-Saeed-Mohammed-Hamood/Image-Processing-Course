import cv2
import numpy as np
import matplotlib.pyplot as plt

# تحميل صورة باللون الرمادي
img = cv2.imread(r"D:\IT\Level 4\Image Processing\Image-Processing-Course\Lab5\HW edge detection\images\test2.png", cv2.IMREAD_GRAYSCALE)

# أقنعة Robinson (8 اتجاهات)
robinson_masks = {
    "North":      np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32),

    "North-East": np.array([[0, 1, 2],
                            [-1, 0, 1],
                            [-2, -1, 0]], dtype=np.float32),

    "East":       np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=np.float32),

    "South-East": np.array([[2, 1, 0],
                            [1, 0, -1],
                            [0, -1, -2]], dtype=np.float32),

    "South":      np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=np.float32),

    "South-West": np.array([[0, -1, -2],
                            [1, 0, -1],
                            [2, 1, 0]], dtype=np.float32),

    "West":       np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32),

    "North-West": np.array([[-2, -1, 0],
                            [-1, 0, 1],
                            [0, 1, 2]], dtype=np.float32),
}

# تطبيق الفلترة لكل اتجاه
results = {}
for direction, kernel in robinson_masks.items():
    results[direction] = cv2.filter2D(img, -1, kernel)

# عرض النتائج
plt.figure(figsize=(15,10))
plt.subplot(3,3,1), plt.imshow(img, cmap='gray'), plt.title("Original")

i = 2
for direction, result in results.items():
    plt.subplot(3,3,i), plt.imshow(result, cmap='gray'), plt.title(direction)
    i += 1

plt.tight_layout()
plt.show()
