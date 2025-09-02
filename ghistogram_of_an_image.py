import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/inzamul/Downloads/tulip.jpg', cv2.IMREAD_GRAYSCALE)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title('Histogram Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.plot(hist, color='black')
plt.show()