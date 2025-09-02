import cv2
import numpy as np
import matplotlib.pyplot as plt 


img = cv2.imread('/home/inzamul/Downloads/tulip.jpg', cv2.COLOR_BGR2RGB)

img_1 = img.copy()
img_1[0:1000, 0:1000] = 0  # Set top-left 1000x1000 pixels to black


plt.figure(figsize=(10,5))
title = ['Original Image', 'Modified Image']
images = [img, img_1]

for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(images[i])
    plt.title(title[i])
    plt.axis('off')
plt.show()