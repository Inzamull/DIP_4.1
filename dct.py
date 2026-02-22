import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import pywt

cwd = os.getcwd()
cwds = cwd.split('/')
img = cv.imread('/home/inzamul/Downloads/tulip new.jpg', 0)


dct = cv.dct(np.float32(img))
LL,(LH, HL, HH) = pywt.dwt2(img,'haar')


threshold = 20


LH_thresh = pywt.threshold(LH, threshold, mode='hard')
HL_thresh = pywt.threshold(HL, threshold, mode='hard')
HH_thresh = pywt.threshold(HH, threshold, mode='hard')

img_reconstructed = pywt.idwt2((LL, (LH_thresh, HL_thresh, HH_thresh)), 'haar') #inverse dwt

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(LH, cmap='gray')
plt.title('LH Before')
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(LH_thresh, cmap='gray')
plt.title('LH After')
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(HL, cmap='gray')
plt.title('HL Before ')
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(HL_thresh, cmap='gray')
plt.title('HL After ')
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(np.clip(img_reconstructed, 0, 255).astype(np.uint8), cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')

plt.tight_layout()
plt.show()