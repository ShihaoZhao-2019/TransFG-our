"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from PIL import Image, ImageDraw
import cv2
import os
from tqdm import tqdm
import numpy as np
import torch
import data
from options.test_options import TestOptions
import models
import random


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

opt = TestOptions().parse()
random_mask_ratio = 1
dataloader = data.create_dataloader(opt)

# model = models.create_model(opt)
# model.eval()
abort_numbers = 0
for i, data_i in tqdm(enumerate(dataloader),desc='Processing', unit='item',position=0,total=len(dataloader)):
    if i * opt.batchSize >= opt.how_many:
        break
    # with torch.no_grad():
    #     generated,_ = model(data_i, mode='inference')
        
    output_path = data_i['output_path']
    image_path = data_i['image_path']
    masks = data_i['mask']
    size = data_i['size']
    rect = data_i['rect']
    # generated = torch.clamp(generated, -1, 1)
    # generated = (generated+1)/2*255
    # generated = generated.cpu().numpy().astype(np.uint8)
    
    masks = torch.clamp(masks, -1, 1)
    masks = masks.cpu().numpy().astype(np.uint8)
    
    for b in range(opt.batchSize):
        
        if(os.path.isdir(os.path.dirname(output_path[b])) == False):
            os.makedirs(os.path.dirname(output_path[b]))
        origin_image = cv2.imread(image_path[b])
        # print('process image... %s' % output_path[b])
        # pred_im_roi = generated[b].transpose((1, 2,  0))[:, :, ::-1]
        # pred_im_roi = cv2.resize(pred_im_roi, (int(size[b]), int(size[b])))
        # pred_image = origin_image.copy()
        # try:
        #     pred_image[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])] = pred_im_roi
        # except:
        #     pass

        mask_roi_gray = masks[b].transpose((1,2,0))[:,:,::-1]
        mask_roi_gray = np.squeeze(mask_roi_gray)
        mask_roi_gray = cv2.resize(mask_roi_gray,(int(size[b]),int(size[b])))
        mask_roi = np.zeros((int(size[b]), int(size[b]), 3))
        mask_roi[:, :, 0] = mask_roi_gray
        mask_roi[:, :, 1] = mask_roi_gray
        mask_roi[:, :, 2] = mask_roi_gray

        masked_image = origin_image.copy()
        x1,y1 = int(rect[0]) , int(rect[1])
        x2,y2 = int(rect[2]) , int(rect[3])
        
        
        origin_file_name, _ = os.path.basename(output_path[b]).split('.')
        origin_image_save_path = output_path[b]

        masked_image_save_path = output_path[b].replace(origin_file_name, origin_file_name)
        
        try:
            masked_image[y1:y2, x1:x2] = masked_image[y1:y2, x1:x2] * mask_roi
            # tmp = masked_image[y1:y1+int(size[b]), x1:x1+int(size[b])]
            # masked_image[y1:y1+int(size[b]), x1:x1+int(size[b])] = tmp * mask_roi
        except:
            print('have problem:',image_path[b], "will be abort, and will save origin image.Abort numbers:",abort_numbers)
            abort_numbers += 1
            cv2.imwrite(origin_image_save_path, origin_image)
            continue

        #添加随机比例划分数据集
        if random.random() <= random_mask_ratio:
           cv2.imwrite(masked_image_save_path, masked_image)
        else:
            cv2.imwrite(origin_image_save_path, origin_image)

