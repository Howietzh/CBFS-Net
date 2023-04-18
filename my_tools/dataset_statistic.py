import cv2, os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#amount of classer
CLASSES_NUM = 5

#find imagee in folder dir
def findImages(dir,topdown=True):
    im_list = []
    if not os.path.exists(dir):
        print("Path for {} not exist!".format(dir))
        raise
    else:
        for root, dirs, files in os.walk(dir, topdown):
            for fl in files:
                im_list.append(fl)
    return im_list

# amount of images corresponding to each classes
images_count = [0]*CLASSES_NUM
# amount of pixels corresponding to each class
class_pixels_count = [0]*CLASSES_NUM
# amount of pixels corresponding to the images of each class
image_pixels_count = [0]*CLASSES_NUM

image_folder = '/mnt/ssd/tzh/Dual-Path-SegNet/data/CCM_original/annotations/train'
im_list = findImages(image_folder)

for im in tqdm(im_list):
    cv_img = cv2.imread(os.path.join(image_folder, im), cv2.IMREAD_GRAYSCALE)
    h, w = cv_img.shape
    for value in range(CLASSES_NUM):
        a = np.where(cv_img==value)
        num_label_pixel = a[0].shape[0]
        images_count[value] += 1
        class_pixels_count[value] += num_label_pixel
        image_pixels_count[value] += h*w

images_count = np.array(images_count)
class_pixels_count = np.array(class_pixels_count)
image_pixels_count = np.array(image_pixels_count)

class_pixels_ratio = class_pixels_count/image_pixels_count

print(images_count)
print(class_pixels_count)
print(image_pixels_count)
print(class_pixels_ratio)

print('number of defect samples is {}'.format(images_count.sum()))
print('ratio of defect pixels is {}'.format(class_pixels_ratio[1:].sum()))

# fig, ax = plt.subplots(figsize=(16, 10), facecolor='white', dpi=80)
# ax.vlines(x=['crack on FPC', 'foreign particals', 'scratch on headface', 'scratch on lens'], ymin=0, ymax=)