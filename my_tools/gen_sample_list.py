import os
import cv2
import numpy as np
from random import shuffle
import random
from tqdm import tqdm

source_dir = 'data/CCM_windowed'

# for sub1 in ('images', 'annotations'):
#     for sub2 in ('train', 'val'):
#         path = os.path.join(target_dir, sub1, sub2)
#         if not os.path.exists(path):
#             os.makedirs(path)
#
# for sub in ('train', 'val'):
#     idx = 0
#     source_img_dir = os.path.join(source_dir, 'images', sub)
#     source_label_dir = os.path.join(source_dir, 'annotations', sub)
#
#     img_list = os.listdir(source_img_dir)
#     for img_name in img_list:
#         img = cv2.imread(os.path.join(source_img_dir, img_name))
#         label = cv2.imread(os.path.join(source_label_dir, img_name), cv2.IMREAD_GRAYSCALE)
#         for h in range(4):
#             for w in range(4):
#                 x1 = 512 * h
#                 x2 = x1 + 512
#                 y1 = 612 * w
#                 y2 = y1 + 612
#                 img_patch = img[x1:x2, y1:y2, :]
#                 label_patch = label[x1:x2, y1:y2]
#                 if np.sum(label_patch) == 0:
#                     cv2.imwrite(os.path.join(target_dir, 'images', sub, str(idx)+'_normal.png'), img_patch)
#                     cv2.imwrite(os.path.join(target_dir, 'annotations', sub, str(idx) + '_normal.png'), label_patch)
#                     idx += 1
#                 else:
#                     cv2.imwrite(os.path.join(target_dir, 'images', sub, str(idx) + '_defect.png'), img_patch)
#                     cv2.imwrite(os.path.join(target_dir, 'annotations', sub, str(idx) + '_defect.png'), label_patch)
#                     idx += 1

for sub in ('train', 'val'):
    source_img_dir = os.path.join(source_dir, 'images', sub)
    source_label_dir = os.path.join(source_dir, 'annotations', sub)
    img_list = os.listdir(source_img_dir)
    normal_list = [name for name in img_list if name.endswith('_normal.png')]
    defect_list = [name for name in img_list if name.endswith('_defect.png')]
    repeat_times = len(normal_list) // len(defect_list)
    over_defect_list = defect_list * repeat_times
    sample_list = normal_list + over_defect_list
    shuffle(sample_list)
    with open(os.path.join(source_dir, sub+'.txt'), 'w') as f:
        for name in tqdm(sample_list):
            line = name.strip('.png') + '\n'
            f.write(line)
    with open(os.path.join(source_dir, sub+'_defect.txt'), 'w') as f:
        for name in tqdm(defect_list):
            line = name.strip('.png') + '\n'
            f.write(line)

