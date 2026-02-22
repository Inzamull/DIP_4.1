import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img_set, titles):
    plt.figure(figsize=(10, 10))
    for i in range(len(img_set)):
        plt.subplot(len(img_set), 2, i * 2 + 1)
        plt.imshow(img_set[i], cmap = 'gray')
        plt.axis('off')
        
        plt.subplot(len(img_set), 2, i * 2 + 2)
        plt.hist(img_set[i].flatten(), 256, [0, 256], color='gray')
    plt.tight_layout()
    plt.show()
    plt.close()

def custom_erode(img, kernel):
    img = img.astype(np.float32)
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)))
    eroded_img = np.zeros_like(img)
    h, w = eroded_img.shape
    for i in range(eroded_img.shape[0]):
        for j in range(eroded_img.shape[1]):
            region = padded_img[i:i+k_h, j:j+k_w]
            masked = region[kernel == 1]
            eroded_img[i, j] = np.min(masked)
    return eroded_img.astype(np.uint8)

def custom_dilate(img, kernel):
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)))
    dilated_img = np.zeros_like(img)
    for i in range(dilated_img.shape[0]):
        for j in range(dilated_img.shape[1]):
            region = padded_img[i:i+k_h, j:j+k_w]
            masked = region[kernel == 1]
            dilated_img[i, j] = np.max(masked)
    return dilated_img.astype(np.uint8)
    
def open(img, kernel):
    return custom_dilate(custom_erode(img, kernel), kernel)

def close(img, kernel):
    return custom_erode(custom_dilate(img, kernel), kernel)
    
def top(img, kernel):
    return cv2.subtract(img, open(img, kernel))

def black(img, kernel):
    return cv2.subtract(close(img, kernel), img)
    
def main():
    img_path = 'nature_flower1.png'
    img = cv2.imread(img_path, 0)
    
    kernel = np.array([[1,1,1],
                       [1,1,1],
                       [1,1,1]])
    
    img_set = [img, cv2.erode(img, kernel), custom_erode(img, kernel), custom_dilate(img, kernel)]
    titles = []
    
    display(img_set, titles)

if __name__ == '__main__':
    main()
    