"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import cv2
import os
from tqdm import tqdm
import numpy as np
import torch
import data
from options.test_options import TestOptions
import random
import concurrent.futures
import copy

opt = TestOptions().parse()
random_mask_ratio = 1
dataloader = data.create_dataloader(opt)

abort_numbers = 0

def add_mask_task(i,data_i,opt):
    if i * opt.batchSize >= opt.how_many:
        return
    output_path = data_i['output_path']
    image_path = data_i['image_path']
    masks = data_i['mask']
    size = data_i['size']
    rect = data_i['rect']
    
    masks = torch.clamp(masks, -1, 1)
    masks = masks.numpy().astype(np.uint8)
    
    for b in range(opt.batchSize):
        
        if(os.path.isdir(os.path.dirname(output_path[b])) == False):
            os.makedirs(os.path.dirname(output_path[b]))
        origin_image = cv2.imread(image_path[b])

        mask_roi_gray = masks[b].transpose((1,2,0))[:,:,::-1]
        mask_roi_gray = np.squeeze(mask_roi_gray)
        mask_roi_gray = cv2.resize(mask_roi_gray,(int(size[b]),int(size[b])))
        mask_roi = np.zeros((int(size[b]), int(size[b]), 3))
        mask_roi[:, :, 0] = mask_roi_gray
        mask_roi[:, :, 1] = mask_roi_gray
        mask_roi[:, :, 2] = mask_roi_gray

        masked_image = origin_image.copy()
        x1,y1 = int(rect[0][b]) , int(rect[1][b])
        x2,y2 = int(rect[2][b]) , int(rect[3][b])
        
        
        origin_file_name, _ = os.path.basename(output_path[b]).split('.')
        origin_image_save_path = output_path[b]

        masked_image_save_path = output_path[b].replace(origin_file_name, origin_file_name)
        
        try:
            masked_image[y1:y2, x1:x2] = masked_image[y1:y2, x1:x2] * mask_roi
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
    
with concurrent.futures.ProcessPoolExecutor(max_workers=opt.nThreads) as executor:
    for i, data_i in tqdm(enumerate(dataloader),desc='Processing', unit='item',position=0,total=len(dataloader)):
        executor.submit(add_mask_task, i,copy.deepcopy(data_i),opt)