from random import shuffle

import cv2, os
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#amount of classer
CLASSES_NUM = 5

#find imagee in folder dir


# amount of images corresponding to each classes
images_count = [0]*CLASSES_NUM
# amount of pixels corresponding to each class
class_pixels_count = [0]*CLASSES_NUM
# amount of pixels corresponding to the images of each class
image_pixels_count = [0]*CLASSES_NUM

image_folder = 'data/CCM_original/annotations/train'
# image_folder = 'data/CCM_windowed/annotations/val'

im_list = os.listdir(image_folder)
# with open(os.path.join('data/CCM_windowed', 'val_defect.txt'), 'r') as f:
#     im_list = f.readlines()

im_list = [im.strip('\n') for im in im_list]
shuffle(im_list)

for im in tqdm(im_list[:5000]):
    cv_img = cv2.imread(os.path.join(image_folder, im), cv2.IMREAD_GRAYSCALE)
    # cv_img = cv2.resize(cv_img, (608, 512))
    h, w = cv_img.shape
    for value in range(CLASSES_NUM):
        a = np.where(cv_img==value)
        num_label_pixel = a[0].shape[0]
        images_count[value] += 1
        class_pixels_count[value] += num_label_pixel
        image_pixels_count[value] += h*w

class_pixels_count = np.array(class_pixels_count)
image_pixels_count = np.array(image_pixels_count)

w = 1/class_pixels_count
print(class_pixels_count/image_pixels_count)
w = w/np.sqrt((np.power(w, 2).sum()))
print(w)
print(w.sum())

