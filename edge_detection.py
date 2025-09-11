import cv2
import matplotlib.pyplot as plt


img = cv2.imread('/home/inzamul/Downloads/rose.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Canny Edge Detection
edges = cv2.Canny(img, 100, 200) 

# Show Results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')

plt.subplot(1,2,2)
plt.title("Canny Edge Detection")
plt.imshow(edges, cmap='gray')
plt.show()
