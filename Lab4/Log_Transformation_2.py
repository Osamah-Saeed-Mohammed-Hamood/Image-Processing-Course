import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/varese.jpg')
img_float = np.float32(img) 
c =  255 / np.log(1 + np.max(img_float))
image_log = c * (np.log(1 + img_float))
image_log = np.uint8(image_log)

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
image_log = cv.cvtColor(image_log, cv.COLOR_BGR2RGB)    

fig , axs = plt.subplots(1,2, figsize=(10,4))
axs[0].imshow(img)
axs[0].set_title('Original Image')
axs[1].imshow(image_log)
axs[1].set_title('Log Transformation')
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()