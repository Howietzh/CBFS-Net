import cv2
import os
import numpy as np

image_root = 'data/CCM_original/images/val'
gt_root = 'data/CCM_original/annotations/val'
deeplabv3plus_root = 'sota/deeplabv3plus_r50-d8/vis'
segformer_root = 'sota/segformer_mit-b1/vis'
icnet_root = 'sota/icnet_r50-d8/vis'
cbbsnet_root = 'backbones/mit-b0/vis'

model_result_path_list = [gt_root, deeplabv3plus_root, segformer_root, cbbsnet_root]

file_list = ['118.png', '271.png', '410.png', '550.png']

result = np.ones((256*4+10*3, 304*7+10*6, 3), dtype=np.uint8) * 255
palette = [[0, 0, 0], [220, 0, 0], [0, 220, 0], [0, 0, 220], [220, 220, 0]]
for i, file in enumerate(file_list):
    ori_image_path = os.path.join(image_root, file)
    ori_image = cv2.imread(ori_image_path)
    gt = cv2.imread(os.path.join(gt_root, file), cv2.IMREAD_GRAYSCALE)
    color_seg = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[gt == label, :] = color
    gt = color_seg
    deeplab = cv2.imread(os.path.join(deeplabv3plus_root, file))
    segformer = cv2.imread(os.path.join(segformer_root, file))
    icnet = cv2.imread(os.path.join(icnet_root, file))
    cbbs = cv2.imread(os.path.join(cbbsnet_root, file))
    x, y, c = np.where(gt!=0)
    h, w, _ = gt.shape
    center_x = (x.min()+x.max())//2
    center_y = (y.min()+y.max())//2
    x1 = center_x - 256//2 if center_x - 256//2 > 0 else 0
    y1 = center_y - 304//2 if center_y - 304//2 > 0 else 0
    x2 = x1 + 256 if x1 + 256 < h else h
    y2 = y1 + 304 if y1 + 304 < w else w
    x1 = x2 - 256
    y1 = y2 - 304
    area = ori_image[x1:x2, y1:y2, :]
    gt = gt[x1:x2, y1:y2, :]
    deeplab = deeplab[x1:x2, y1:y2, :]
    deeplab = cv2.cvtColor(deeplab, cv2.COLOR_RGB2BGR)
    segformer = segformer[x1:x2, y1:y2, :]
    segformer = cv2.cvtColor(segformer, cv2.COLOR_RGB2BGR)
    icnet = icnet[x1:x2, y1:y2, :]
    icnet = cv2.cvtColor(icnet, cv2.COLOR_RGB2BGR)
    cbbs = cbbs[x1:x2, y1:y2, :]
    cbbs = cv2.cvtColor(cbbs, cv2.COLOR_RGB2BGR)
    ori_image = cv2.resize(ori_image, (304, 256))
    result[i*(256+10):i*(256+10)+256, 0:304, :] = ori_image
    result[i * (256 + 10):i * (256 + 10) + 256, 314:618, :] = area
    result[i * (256 + 10):i * (256 + 10) + 256, 628:932, :] = gt
    result[i * (256 + 10):i * (256 + 10) + 256, 942:1246, :] = deeplab
    result[i * (256 + 10):i * (256 + 10) + 256, 1256:1560, :] = segformer
    result[i * (256 + 10):i * (256 + 10) + 256, 1570:1874, :] = icnet
    result[i * (256 + 10):i * (256 + 10) + 256, 1884:2188, :] = cbbs

cv2.imwrite('experiment_images/qualitative_exp.png', result)




