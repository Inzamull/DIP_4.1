import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = '/home/inzamul/Downloads/paddy_f_1.jpg' 
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


bit_planes = []

for i in range(8):
    bit_plane = ((img >> i) & 1) * 255
    bit_planes.append(bit_plane)

plt.figure(figsize=(12, 6))
plt.subplot(2, 5, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

for i in range(8):
    plt.subplot(2, 5, i+2)
    plt.imshow(bit_planes[i], cmap='gray')
    plt.title(f'Bit Plane {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()

