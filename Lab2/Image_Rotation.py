import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("images/balloons.jpg")

img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)

center = (img_rgb.shape[1] // 2,img_rgb.shape[0] // 2)

rotation_matrix = cv.getRotationMatrix2D(center,30,0.7)
rotation_matrix2 = cv.getRotationMatrix2D(center,30,1)

rotated_image = cv.warpAffine(img_rgb,rotation_matrix,(img.shape[1],img.shape[0]))
rotated_image2 = cv.warpAffine(img_rgb,rotation_matrix2,(img.shape[1],img.shape[0]))

fig , axs = plt.subplots(1,3,figsize = (14,4))

axs[0].imshow(img_rgb)
axs[0].set_title('Origin Image')
axs[1].imshow(rotated_image)
axs[1].set_title('Image Rotation')
axs[2].imshow(rotated_image2)
axs[2].set_title('Image Rotation2')

plt.show()
