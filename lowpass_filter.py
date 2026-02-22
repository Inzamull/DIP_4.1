import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/home/inzamul/Downloads/rose.jpg', 0)
dft = np.fft.fft2(img)
shifted_dft = np.fft.fftshift(dft)

def create_filter(height, width, center, radius):
    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    return mask

rows, cols = img.shape
print(f"Image Size: Height={rows}, Width={cols}")

center_x = cols // 2
center_y = rows // 2
filter_radius = 100

lowpass_filter = create_filter(
    height=rows,
    width=cols,
    center=(center_x, center_y),
    radius=filter_radius
)


result = shifted_dft * lowpass_filter



unshifted_dft = np.fft.ifftshift(result)

filtered_image_complex = np.fft.ifft2(unshifted_dft)


final_image = np.abs(filtered_image_complex)



plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(lowpass_filter, cmap='gray')
plt.title("Low-Pass Filter Mask")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(final_image, cmap='gray')
plt.title("Filtered (Blurred) Image")
plt.axis('off')

plt.tight_layout()
plt.show()