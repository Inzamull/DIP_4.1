import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('/home/inzamul/Downloads/rose.jpg')

va_mat = np.matrix([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])

hor_mat = np.matrix([[-1,-2,-1],
                     [0,0,0],
                     [1,2,1]])



va_mat2 = np.array([[1,2,3],
                    [4,5,6],
                    [7,8,9]])/9


hor_mat2 = np.array([[9,8,7],
                     [6,5,4],
                     [3,2,1]])/9






cv2.filter2D(img, -1, va_mat)
cv2.filter2D(img, -1, hor_mat)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Image')


plt.subplot(1, 3, 2)
plt.imshow(cv2.filter2D(img, -1, va_mat))
plt.axis('off')
plt.title('Vertical Sobel Filter')

plt.subplot(1, 3, 3)
plt.imshow(cv2.filter2D(img, -1, hor_mat))
plt.axis('off') 
plt.title('Horizontal Sobel Filter')

plt.subplot(1,3,1)
plt.imshow(cv2.filter2D(img, -1, va_mat2))
plt.axis('off')
plt.title('Vertical Sobel Filter 2')

plt.show()