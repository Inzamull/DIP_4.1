import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('/home/inzamul/Downloads/rose.jpg', 0)

#Smoothing (Average) Kernel
kernel_avg = np.ones((3,3), np.float32)/9
smooth = cv2.filter2D(img, -1, kernel_avg)

#Sobel Kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
sobelx_img = cv2.filter2D(img, -1, sobel_x)
sobely_img = cv2.filter2D(img, -1, sobel_y)

# Prewitt Kernels
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_y = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]])
prewittx_img = cv2.filter2D(img, -1, prewitt_x)
prewitty_img = cv2.filter2D(img, -1, prewitt_y)

#Laplacian Kernel
laplace_kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])
laplace_img = cv2.filter2D(img, -1, laplace_kernel)


#Display Image
titles = ['Original', 'Smoothing', 
          'Sobel X', 'Sobel Y', 
          'Prewitt X', 'Prewitt Y', 
          'Laplacian']

images = [img, smooth, 
          sobelx_img, sobely_img, 
          prewittx_img, prewitty_img, 
          laplace_img]

plt.figure(figsize=(12, 8))
for i in range(7):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()