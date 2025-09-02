import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('/home/inzamul/Downloads/tulip.jpg', cv2.IMREAD_GRAYSCALE)


mean = 0
std = 80  # noise strength
gaussian = np.random.normal(mean, std, img.shape).astype(np.float32)


noisy_img = cv2.add(img.astype(np.float32), gaussian)
#noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(1,2,2), plt.imshow(noisy_img, cmap='gray'), plt.title("Gaussian Noise")
plt.show()
