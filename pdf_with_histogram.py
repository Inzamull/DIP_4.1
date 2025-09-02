import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('/home/inzamul/Downloads/tulip.jpg', 0)


pixels = img.flatten()


hist, bins = np.histogram(pixels, bins=256, range=[0,256])

pdf = hist / hist.sum()

cdf = pdf.cumsum()







img = np.array([[2, 2, 3],
                [3, 4, 4],
                [4, 5, 5]], dtype=np.uint8)


L = 8 
hist = np.zeros(L, dtype=int)

for r in img.flatten():
    hist[r] += 1

# PDF
total_pixels = img.size
pdf = hist / total_pixels

# CDF
cdf = np.cumsum(pdf)


equalized_mapping = np.round((L-1) * cdf).astype(np.uint8)
eq_img = equalized_mapping[img]


print("Original Image:\n", img)
print("Histogram:\n", hist)
print("PDF:\n", pdf)
print("CDF:\n", cdf)
print("Equalized Mapping:\n", equalized_mapping)
print("Equalized Image:\n", eq_img)


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plt(np.arange(L), hist, color='blue')
plt.title("Original Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
plt.plot(np.arange(L), np.bincount(eq_img.flatten(), minlength=L), color='red')
plt.title("Equalized Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")





#plt.subplot(1,2,1); plt.plot(pdf, color='blue'); plt.title("PDF")
#plt.subplot(1,2,2); plt.plot(cdf, color='red'); plt.title("CDF")
plt.show()
