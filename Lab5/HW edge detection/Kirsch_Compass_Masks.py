import cv2
import numpy as np
import matplotlib.pyplot as plt

# تحميل صورة باللون الرمادي
img = cv2.imread(r"D:\IT\Level 4\Image Processing\Image-Processing-Course\Lab5\HW edge detection\images\test2.png", cv2.IMREAD_GRAYSCALE)

# أقنعة Kirsch (8 اتجاهات)
kirsch_masks = {
    "North":      np.array([[-3, -3, -3],
                            [-3,  0, -3],
                            [ 5,  5,  5]], dtype=np.float32),

    "North-East": np.array([[-3, -3, -3],
                            [ 5,  0, -3],
                            [ 5,  5, -3]], dtype=np.float32),

    "East":       np.array([[-3, -3,  5],
                            [-3,  0,  5],
                            [-3, -3,  5]], dtype=np.float32),

    "South-East": np.array([[-3,  5,  5],
                            [-3,  0,  5],
                            [-3, -3, -3]], dtype=np.float32),

    "South":      np.array([[ 5,  5,  5],
                            [-3,  0, -3],
                            [-3, -3, -3]], dtype=np.float32),

    "South-West": np.array([[ 5,  5, -3],
                            [ 5,  0, -3],
                            [-3, -3, -3]], dtype=np.float32),

    "West":       np.array([[ 5, -3, -3],
                            [ 5,  0, -3],
                            [ 5, -3, -3]], dtype=np.float32),

    "North-West": np.array([[-3, -3, -3],
                            [-3,  0,  5],
                            [-3,  5,  5]], dtype=np.float32),
}

# تطبيق الفلترة لكل اتجاه
results = {}
for direction, kernel in kirsch_masks.items():
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
