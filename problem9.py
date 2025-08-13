import matplotlib.pyplot as plt
import numpy as np
import cv2


def main():
    img_path = "/home/inzamul/Downloads/rgb_image.png"
    img_3D = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_3D, cv2.COLOR_BGR2RGB)
    

    img_red = img_rgb[:, :, 0]
    img_green = img_rgb[:, :, 1]
    img_blue = img_rgb[:, :, 2]


    plt.figure(figsize=(25,22))
    plt.subplot(4,2,1)
    plt.imshow(img_rgb)  

    plt.subplot(4,2,2)
    plt.imshow(img_red, cmap = "Reds")
    
    plt.subplot(4,2,3)
    plt.imshow(img_green, cmap = 'Greens')
    
    
    plt.subplot(4,2,4)
    plt.imshow(img_blue, cmap = 'Blues')
    
    
    red_histogram = histogram(img_red)
    green_histogram = histogram(img_green)
    blue_histogram = histogram(img_blue)

    plt.subplot(4,2,5)
    plt.plot(range(256),red_histogram, color='red')
    
    plt.subplot(4,2,6)
    plt.plot(range(256), green_histogram, color = 'green')
    
    plt.subplot(4,2,7)
    plt.plot(range(256), blue_histogram, color = 'blue')
    

    plt.tight_layout()
    plt.show()
    plt.close()


    

def histogram(img_2D):
    h, w = img_2D.shape
    hist = np.zeros(256, dtype = int)  


    for i in range(h):
        for j in range(w):
            pixel_value = img_2D[i,j]
            hist[pixel_value] += 1

    print(hist)
    return hist


if __name__ == "__main__":
    main()