import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# ================= Custom filter function =================
def custom_filter2D(img, kernel, mode='same'):
    kernel = np.array(kernel, dtype=np.float32)
    k_h, k_w = kernel.shape
    flipped_kernel = np.flipud(np.fliplr(kernel))

    if img.ndim == 3:  # color image
        channels = []
        for c in range(img.shape[2]):
            filtered = custom_filter2D(img[:, :, c], kernel, mode)
            channels.append(filtered)
        return np.stack(channels, axis=2)

    img = img.astype(np.float32)
    img_h, img_w = img.shape

    if mode == 'same':
        pad_h, pad_w = k_h // 2, k_w // 2
        padded_img = np.zeros((img_h + 2*pad_h, img_w + 2*pad_w), dtype=np.float32)
        padded_img[pad_h:pad_h+img_h, pad_w:pad_w+img_w] = img
    else:
        padded_img = img
        pad_h, pad_w = 0, 0

    out_h = img_h if mode == 'same' else img_h - k_h + 1
    out_w = img_w if mode == 'same' else img_w - k_w + 1
    output = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            region = padded_img[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(region * flipped_kernel)

    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
    return output.astype(np.uint8)

# ================= Display function =================
def display_img(img_set, titles, row, col, max_width=300):
    plt.figure(figsize=(15, 10))
    for k, img in enumerate(img_set):
        plt.subplot(row, col, k+1)

        # resize for plotting
        scale_ratio = max_width / img.shape[1]
        new_height = int(img.shape[0] * scale_ratio)
        if img.ndim == 2:
            img_plot = cv2.resize(img, (max_width, new_height))
            plt.imshow(img_plot, cmap='gray')
        else:
            img_plot = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (max_width, new_height))
            plt.imshow(img_plot)

        plt.title(titles[k])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ================= Main function =================
def main():
    uploaded = files.upload()
    img_path = list(uploaded.keys())[0]
    img = cv2.imread(img_path)

    if img is None:
        print("Error: Image not loaded!")
        return

    # resize large image to width=800
    max_width = 800
    scale_ratio = max_width / img.shape[1]
    img = cv2.resize(img, (max_width, int(img.shape[0]*scale_ratio)))

    # ================= Define kernels =================
    kernel_vertical = [[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]]

    kernel_horizontal = [[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]]

    kernel_average = [[1/9, 1/9, 1/9],
                      [1/9, 1/9, 1/9],
                      [1/9, 1/9, 1/9]]

    # ================= Apply filters =================
    filtered_img = [
        img,
        custom_filter2D(img, kernel_vertical),
        custom_filter2D(img, kernel_horizontal),
        custom_filter2D(img, kernel_average)
    ]

    titles = ['Original', 'Vertical Sobel', 'Horizontal Sobel', 'Average Filter']

    display_img(filtered_img, titles, row=2, col=2)

if __name__ == '__main__':
    main()
