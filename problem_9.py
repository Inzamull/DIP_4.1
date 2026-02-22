import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

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

def adjust_contrust(img, mode):
    img = img.astype(np.float32)
    if mode == 'low':
        img = img * 0.5 + 64
    elif mode == 'high':
        img = (img - 128) * 2 + 128
    return np.clip(img, 0, 255).astype(np.uint8)

def hist_matching_cdf(src, ref):
    src_hist, _ = np.histogram(src.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref.flatten(), 256, [0, 256])
    
    src_cdf = src_hist.cumsum() / src_hist.sum()
    ref_cdf = ref_hist.cumsum() / ref_hist.sum()
    
    mapping = np.zeros(256, dtype = np.uint8)
    for i in range(256):
        diff = np.abs(ref_cdf - src_cdf[i])
        mapping[i] = np.argmin(diff)
    
    matched = cv2.LUT(src, mapping)
    return matched

def hist_matching_spec(src, ref):
    src_hist, _ = np.histogram(src.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref.flatten(), 256, [0, 256])
    
    src_cdf = src_hist.cumsum() / src_hist.sum()
    ref_cdf = ref_hist.cumsum() / ref_hist.sum()
    
    mapping = np.zeros(256, dtype = np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and ref_cdf[j] < src_cdf[i]:
            j += 1
        mapping[i] = j
    
    matched = cv2.LUT(src, mapping)
    return matched

def main():
    img_path = 'nature1.png'
    img = cv2.imread(img_path, 0)
    
    src_low = adjust_contrust(img, 'low')
    ref_high = adjust_contrust(img, 'high')
    
    #img_cdf = hist_matching_cdf(src_low, ref_high)
    matched_builtin = match_histograms(source_low, reference_high)
    
    img_set = [img, src_low, ref_high, hist_matching_cdf(src_low, ref_high), hist_matching_spec(src_low, ref_high), matched_builtin]
    titles = []
    
    display(img_set, titles)

if __name__ == '__main__':
    main()