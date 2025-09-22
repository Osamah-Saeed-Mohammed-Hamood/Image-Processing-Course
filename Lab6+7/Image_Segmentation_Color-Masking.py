import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

sample_image = cv.imread('images/shapes.webp')
img = cv.cvtColor(sample_image, cv.COLOR_BGR2RGB)

low = np.array([0, 0, 0])
high = np.array([215, 80, 80])

mask = cv.inRange(img, low, high)
result = cv.bitwise_and(img, img, mask=mask)
fig ,axes = plt.subplots(1,3, figsize=(15,5))
axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Mask')
axes[1].axis('off')
axes[2].imshow(result)
axes[2].set_title('Result Image')
axes[2].axis('off')
plt.show()