import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('/home/inzamul/Downloads/image_of_i', cv2.IMREAD_GRAYSCALE)


ker = np.array([[1,1,1],
                [1,1,1],
                [1,1,1]])


ero = cv2.erode(img,ker,iterations=1)

dil = cv2.dilate(img,ker,iterations=1)




plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')

plt.subplot(1,2,2)
plt.title("Canny Edge Detection")
plt.imshow(ero, cmap='gray')
plt.show()
