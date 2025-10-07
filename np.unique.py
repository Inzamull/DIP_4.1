import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread('/home/inzamul/Downloads/rose.jpg',0)


u,c = np.unique(img, return_counts= True)

print(u)
print(c)
length = len(u)
print("Length of u:", length)
