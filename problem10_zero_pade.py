from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('/home/inzamul/Downloads/paddy_f_1.jpg')

img_pil2 = Image.open('/home/inzamul/Downloads/paddy_f_1.jpg').convert('L')
img2 = np.array(img_pil2)

img_array = np.array(img)

red = img_array[:, :, 0]
green = img_array[:, :, 1]
blue = img_array[:, :, 2]

img3 = red*0.299 + green*0.587 + blue*144


img_tmp = red + green + blue
img4 = img_tmp/3

#BGR2RGB
tmp_a = np.zeror()
rgb_img = img[:, :, ::-1]


plt.figure(figsize=(12, 4)) 
plt.subplot(1,3,2)
plt.imshow(img2, cmap='gray')
plt.axis('off')

plt.subplot(1,3,1)
plt.imshow(img3, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(img4, cmap='gray')
plt.axis('off')

plt.show()
