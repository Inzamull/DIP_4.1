import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/home/inzamul/Downloads/rose.jpg', 0)

equalized = cv2.equalizeHist(img)

# 3. Show images
plt.figure(figsize=(10,5))

plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')

plt.subplot(2,2,2)
plt.title("Equalized Image")
plt.imshow(equalized, cmap='gray')

#Histogram before
plt.subplot(2,2,3)
plt.title("Original Histogram")
plt.hist(img.ravel(), 256, [0,256])

#Histogram after
plt.subplot(2,2,4)
plt.title("Equalized Histogram")
plt.hist(equalized.ravel(), 256, [0,256])

plt.tight_layout()
plt.show()
