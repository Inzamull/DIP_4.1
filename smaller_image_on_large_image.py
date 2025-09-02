import cv2
import numpy as np
import matplotlib.pyplot as plt


big = cv2.imread('/home/inzamul/Downloads/big_image.jpg')
small = cv2.imread('/home/inzamul/Downloads/rose.jpg')



result = np.zeros_like(big)


H, W = big.shape[:2]      
h, w = small.shape[:2]    


y_start = (H - h) // 2
x_start = (W - w) // 2


result[y_start:y_start+h, x_start:x_start+w] = small




plt.figure(figsize=(12,5))
title = ['Big Image', 'Small Image', 'Result (Small at Center)']
images = [big, small, result]   

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(title[i])
    plt.axis('off')
plt.show()


