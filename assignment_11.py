import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('/home/inzamul/Downloads/rose.jpg', cv2.IMREAD_GRAYSCALE)


he_img = cv2.equalizeHist(img)


tile_size = 64
h, w = img.shape
ahe_img = np.zeros_like(img)


for i in range(0, h, tile_size):
    for j in range(0, w, tile_size):
        tile = img[i:i+tile_size, j:j+tile_size]
        eq_tile = cv2.equalizeHist(tile)
        ahe_img[i:i+tile_size, j:j+tile_size] = eq_tile


plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Histogram Equalization (HE)")
plt.imshow(he_img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Adaptive Histogram Equalization (AHE)")
plt.imshow(ahe_img, cmap='gray')
plt.axis('off')

plt.show()
