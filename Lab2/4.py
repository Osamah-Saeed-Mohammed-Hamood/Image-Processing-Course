import cv2 as cv
import matplotlib.pyplot as plt 
import numpy as np

img = cv.imread("images/balloons.jpg")
img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
width = img_rgb.shape[1]
height = img_rgb.shape[0]
tx = 100
ty = 70

translation_matrix = np.array([[1,0,tx],[0,1,ty]],dtype=np.float32)

translated_img = cv.warpAffine(img_rgb,translation_matrix,(width,height))

fig , axs = plt.subplots(1,2,figsize = (7,4))
axs[0].imshow(img_rgb)
axs[0].set_title('Original Image')
axs[1].imshow(translated_img)
axs[1].set_title("Image Translation")

plt.show()