import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('/home/inzamul/Downloads/rose.jpg', 0)
h, w = img.shape


slice_height = h // 10

parts = []

plt.figure(figsize=(20,20))

for i in range(10):
    part = img[i*slice_height:(i+1)*slice_height, :]
    parts.append(part)

    plt.subplot(5,2,i+1)   
    plt.imshow(part, cmap='gray')
    plt.title(f"Slice {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()


reconstructed = np.concatenate(parts, axis=0)


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Reconstructed (Concatenate)")
plt.imshow(reconstructed, cmap='gray')
plt.axis('off')

plt.show()
