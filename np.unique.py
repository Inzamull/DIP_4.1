import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread('/home/inzamul/Downloads/rose.jpg', 0)


dft = np.fft.fft2(img)


shifted_dft = np.fft.fftshift(dft)

magnitude_dft = np.log(np.abs(shifted_dft) + 1)


diff_dft = np.abs(shifted_dft - magnitude_dft)
sum_dft = np.abs(shifted_dft + magnitude_dft)


plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)
plt.title("Shifted DFT (Magnitude Spectrum)")
plt.imshow(np.abs(shifted_dft), cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Log Magnitude Spectrum")
plt.imshow(magnitude_dft, cmap='gray')
plt.axis('off')

plt.subplot(2,2,1)
plt.title("DFT Difference")
plt.imshow(np.log(diff_dft + 1), cmap='gray')

plt.subplot(2,2,2)
plt.title("DFT Sum")
plt.imshow(np.log(sum_dft + 1), cmap='gray')


plt.show()
