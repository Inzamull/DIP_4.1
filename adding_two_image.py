import cv2
import matplotlib.pyplot as plt
import numpy as np


img_1 = cv2.imread('/home/inzamul/Downloads/tulip.jpg', cv2.COLOR_BGR2RGB)
img_2 = cv2.imread('/home/inzamul/Downloads/rose.jpg', cv2.COLOR_BGR2RGB)


img_2 = cv2.resize(img_2,(img_1.shape[1], img_1.shape[0]))



result = cv2.add(img_1, img_2)


plt.figure(figsize=(12,5))
title = ['Image 1', 'Image 2', 'Added Image']
images = [img_1, img_2, result]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i])
    plt.title(title[i])
    plt.axis('off')
plt.show()
