import numpy as np
from PIL import Image
import os
import cv2

original_image = np.array(Image.open('data/train/10_left.jpeg'))

image = np.array(original_image)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = image_gray > 0.1 * np.mean(image_gray[image_gray != 0])

row_sums = np.sum(img, axis=1)
col_sums = np.sum(img, axis=0)

rows = np.where(row_sums > img.shape[1]*0.02)
cols = np.where(col_sums > img.shape[0]*0.02)

print(rows)