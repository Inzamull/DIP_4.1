import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/home/inzamul/Downloads/rose.jpg', 0)
dft = np.fft.fft2(img)
shifted_dft = np.fft.fftshift(dft)



rows, cols = img.shape
print(f"Image Size: Height={rows}, Width={cols}")

center_x = cols // 2
center_y = rows // 2


u = np.arange(rows)
v = np.arange(cols)

U,V = np.meshgrid(u - center_y, v - center_x, indexing='ij')

D = np.sqrt(U**2 + V**2)
D1 = 40
n = 2

butterworth_lpf = 1/(1 + (D/D1)**(2*n))
butterworth_hpf = 1 - butterworth_lpf

result = shifted_dft * butterworth_lpf
unshifted_dft = np.fft.ifftshift(result)
filtered_image_complex = np.fft.ifft2(unshifted_dft)
final_image = np.abs(filtered_image_complex)

result1 = shifted_dft * butterworth_hpf
unshifted_dft1 = np.fft.ifftshift(result1)
filtered_image_complex1 = np.fft.ifft2(unshifted_dft1)
final_image1 = np.abs(filtered_image_complex1)


plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(final_image, cmap='gray')
plt.title("Butterworth Low pass")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(final_image1, cmap='gray')
plt.title("Butterworth high pass")
plt.axis('off')

plt.tight_layout()
plt.show()

