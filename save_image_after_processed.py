import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    big = cv2.imread('/home/inzamul/Downloads/big_image.jpg')
    print(big.shape)
    

    processed_img_pre = 255 - big
    processed_img = cv2.cvtColor(big,cv2.COLOR_BGR2RGB)

    #save image
    save_img_path = '/home/inzamul/Downloads/processed_image.jpg'
    cv2.imwrite(save_img_path, processed_img)
    print(processed_img.shape)


    #save figure
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(processed_img)
    plt.title('Processed Image')
    plt.axis('off')
    plt.savefig('/home/inzamul/Downloads/comparison_image.jpg')
    plt.show()

    #save matrix
    np_file = '/home/inzamul/Downloads/image_matrix.npy'
    np.save(np_file, big, processed_img)
    #np.savez_compressed()
   
    #open npy file
    img_set = np.load(np_file)
    print(len(img_set) , img_set['image_matrix'])
    


   
if __name__ == "__main__":
    main()