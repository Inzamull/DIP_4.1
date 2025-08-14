import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = "/home/inzamul/Downloads/tulip.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


img = cv2.resize(img, (400, 300))


def log_transform(image):
    img_norm = image / 255.0
    c = 255 / np.log(1 + np.max(img_norm))
    log_img = c * np.log(1 + img_norm)
    return np.uint8(log_img)

def gamma_transform(image, gamma):
    img_norm = image / 255.0
    gamma_img = np.power(img_norm, gamma) * 255
    return np.uint8(gamma_img)

log_img = log_transform(img)
gamma_img = gamma_transform(img, 0.5)  # gamma < 1 brightens


threshold_values = [60, 120, 180]


def plot_thresholds(image, title_prefix):
    fig, axes = plt.subplots(len(threshold_values), 2, figsize=(8, 8))
    fig.suptitle(f"{title_prefix} - Thresholding Results", fontsize=14)

    for idx, T in enumerate(threshold_values):
        _, binary_img = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)

        
        axes[idx, 0].imshow(binary_img, cmap='gray')
        axes[idx, 0].set_title(f"T = {T}")
        axes[idx, 0].axis('off')

        
        axes[idx, 1].hist(image.ravel(), bins=256, range=(0, 256), color='black')
        axes[idx, 1].axvline(T, color='red', linestyle='--', label='Threshold')
        axes[idx, 1].legend()
        axes[idx, 1].set_title("Histogram")

    plt.tight_layout()
    plt.show()


plot_thresholds(img, "Original Image")
plot_thresholds(log_img, "Logarithmic Transformation")
plot_thresholds(gamma_img, "Gamma Transformation (γ=0.5)")
