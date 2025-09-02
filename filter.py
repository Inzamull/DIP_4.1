import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
   
    img_gray = cv2.imread('/home/inzamul/Downloads/tulip.jpg', 0)

   
    row, col = img_gray.shape
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy_img = img_gray + gauss * 255  
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    
    avg_filter = np.ones((3,3), np.float32) / 9

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]], dtype=np.float32)

    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)

    prewitt_y = np.array([[-1, -1, -1],
                          [0,  0,  0],
                          [1,  1,  1]], dtype=np.float32)

    laplace_4 = np.array([[ 0, -1,  0],
                          [-1,  4, -1],
                          [ 0, -1,  0]], dtype=np.float32)

    laplace_8 = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=np.float32)

    
    avg = cv2.filter2D(noisy_img, -1, avg_filter)
    sobel_x_f = cv2.filter2D(noisy_img, -1, sobel_x)
    sobel_y_f = cv2.filter2D(noisy_img, -1, sobel_y)
    prewitt_x_f = cv2.filter2D(noisy_img, -1, prewitt_x)
    prewitt_y_f = cv2.filter2D(noisy_img, -1, prewitt_y)
    laplace_4_f = cv2.filter2D(noisy_img, -1, laplace_4)
    laplace_8_f = cv2.filter2D(noisy_img, -1, laplace_8)

  
    img_set = [img_gray, noisy_img, avg, sobel_x_f, sobel_y_f,
               prewitt_x_f, prewitt_y_f, laplace_4_f, laplace_8_f]
    img_title = ['Original', 'Gaussian Noise', 'Avg filter',
                 'Sobel-X', 'Sobel-Y',
                 'Prewitt-X', 'Prewitt-Y',
                 'Laplace-4', 'Laplace-8']

    display(img_set, img_title)

def display(img_set, img_title):
    plt.figure(figsize=(18,12))
    for i in range(len(img_set)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(img_title[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
