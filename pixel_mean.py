import torch
import torch.nn as nn
import torch.nn.functional as F
 
import sys
import math
import numpy as np
import os
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.datasets.folder import default_loader
import cv2
from os.path import join
import scipy
import pandas as pd

import scipy
from scipy import io
# CUB
# =============================================================================
# root = 'D:/study/Weakly_Supervised_Deep_Detection_Networks/WSDDN.pytorch-master4/CUB'
# img_txt_file = open(os.path.join(root, 'images.txt'))
# img_name_list = []
# for line in img_txt_file:
#     img_name_list.append(line[:-1].split(' ')[-1])
# print(img_name_list[0])
# 
# i = 0
# img_sum = 0
# total_pixel = 0
# sum_R = 0
# sum_G = 0
# sum_B = 0
# for img_file in img_name_list:
#     img = imageio.imread(os.path.join(root, 'images', img_file))
#     if len(img.shape) == 2:
#         img = np.stack([img] * 3, 2)
#     img = np.array(img)
#     print(i)
#     print(img.shape)
#     num_pixel = img.shape[0]*img.shape[1]
#     print('num_pixel',num_pixel)
#     
# 
#     sum_pixel = np.sum(img,axis = 0)
#     print('sum_pixel0', sum_pixel.shape)
#     sum_pixel = np.sum(sum_pixel,axis = 0)
#     print('sum_pixel1', sum_pixel.shape)
#     print('sum_pixel1', sum_pixel )
#     
#     R = sum_pixel[0]
#     G = sum_pixel[1]
#     B = sum_pixel[2]
#     
#     print('RGB', R,G,B )
#     
#     print('img_sum0',img_sum)
#     img_sum += sum_pixel
#     print('img_sum1',img_sum)
#     total_pixel += num_pixel
#     
#     print('sum_RGB',sum_R,sum_G,sum_B )
#     sum_R += R
#     sum_G += G
#     sum_B += B
#     print('sum_RGB',sum_R,sum_G,sum_B )
#     i += 1
#     
#     '''
#     # if i ==2:
#     if i ==191:
#         break
#     '''
#     
# print('\n')
# print('img_sum',img_sum)
# print('total_pixel',total_pixel)
# mean = img_sum/total_pixel
# print('mean',mean)
# 
# mean_R = sum_R/total_pixel
# mean_G = sum_G/total_pixel
# mean_B= sum_B/total_pixel
# print('mean_RGB',mean_R,mean_G,mean_B )
# =============================================================================

# =============================================================================
# # DOG
# def load_split(root):
#     if train:
#         split = scipy.io.loadmat(join(root, 'train_list.mat'))['annotation_list']
#         labels = scipy.io.loadmat(join(root, 'train_list.mat'))['labels']
#     else:
#         split = scipy.io.loadmat(join(root, 'test_list.mat'))['annotation_list']
#         labels = scipy.io.loadmat(join(root, 'test_list.mat'))['labels']
# 
#     split = [item[0][0] for item in split]
#     labels = [item[0]-1 for item in labels]
#     return list(zip(split, labels))
# 
# # train = True
# train = False
# root = 'D:/study/dog'
# split = load_split(root)
# images_folder = join(root, 'Images')
# 
# _breed_images = [(annotation+'.jpg', idx) for annotation, idx in split]
# _flat_breed_images = _breed_images
# print(len(_flat_breed_images))
# 
# 
# 
# i = 0
# img_sum = 0
# total_pixel = 0
# sum_R = 0
# sum_G = 0
# sum_B = 0
# for index in range(len(_flat_breed_images)):
#     image_name, target_class = _flat_breed_images[index]
#     image_path = join(images_folder, image_name)
#     # print(image_name)
#     img = Image.open(image_path).convert('RGB')       # RGB
#     img = np.asarray(img)
# # =============================================================================
# #     img = imageio.imread(image_path)
# # =============================================================================
#     
#     print(i)
#     print(img.shape)
#     num_pixel = img.shape[0]*img.shape[1]
#     print('num_pixel',num_pixel)
#     
#  
#     sum_pixel = np.sum(img,axis = 0)
#     print('sum_pixel0', sum_pixel.shape)
#     sum_pixel = np.sum(sum_pixel,axis = 0)
#     print('sum_pixel1', sum_pixel.shape)
#     print('sum_pixel1', sum_pixel )
#     
#     R = sum_pixel[0]
#     G = sum_pixel[1]
#     B = sum_pixel[2]
#     
#     print('RGB', R,G,B )
#     
#     print('img_sum0',img_sum)
#     img_sum += sum_pixel
#     print('img_sum1',img_sum)
#     total_pixel += num_pixel
#     
#     print('sum_RGB',sum_R,sum_G,sum_B )
#     sum_R += R
#     sum_G += G
#     sum_B += B
#     print('sum_RGB',sum_R,sum_G,sum_B )
#     i += 1
#     
#     '''
#     # if i ==2:
#     if i ==191:
#         break
#     '''
#     
# print('\n')
# print('img_sum',img_sum)
# print('total_pixel',total_pixel)
# mean = img_sum/total_pixel
# print('mean',mean)
# 
# mean_R = sum_R/total_pixel
# mean_G = sum_G/total_pixel
# mean_B= sum_B/total_pixel
# print('mean_RGB',mean_R,mean_G,mean_B )
# =============================================================================



# nabirds
# =============================================================================
# def get_continuous_class_map(class_labels):
#     label_set = set(class_labels)
#     return {k: i for i, k in enumerate(label_set)}
# 
# def load_class_names(dataset_path=''):
#     names = {}
# 
#     with open(os.path.join(dataset_path, 'classes.txt')) as f:
#         for line in f:
#             pieces = line.strip().split()
#             class_id = pieces[0]
#             names[class_id] = ' '.join(pieces[1:])
# 
#     return names
# 
# def load_hierarchy(dataset_path=''):
#     parents = {}
# 
#     with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
#         for line in f:
#             pieces = line.strip().split()
#             child_id, parent_id = pieces
#             parents[child_id] = parent_id
# 
#     return parents
# 
# 
# root = '/data/kb/tanyuanyong/TransFG-master/data/nabirds'
# dataset_path = os.path.join(root, 'nabirds')
# loader = default_loader
# base_folder = 'nabirds/images'
# 
# image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
#                                   sep=' ', names=['img_id', 'filepath'])
# # len 48562
# image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
#                                          sep=' ', names=['img_id', 'target'])
# # len 555
# label_map = get_continuous_class_map(image_class_labels['target'])
# # print('label_map',label_map)
# train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
#                                sep=' ', names=['img_id', 'is_training_img'])
# data = image_paths.merge(image_class_labels, on='img_id')
# data = data.merge(train_test_split, on='img_id')
# class_names = load_class_names(dataset_path)
# class_hierarchy = load_hierarchy(dataset_path)
# 
# print(len(image_paths),'\n')
# 
# i = 0
# img_sum = 0
# total_pixel = 0
# sum_R = 0
# sum_G = 0
# sum_B = 0
# for index in range(len(image_paths)):
#     sample = data.iloc[index]
#     path = os.path.join(root, base_folder, sample.filepath)
#     target = label_map[sample.target]
#     img = loader(path)
# 
#     img = np.array(img)
# # =============================================================================
# #     img = imageio.imread(image_path)
# # =============================================================================
#     
#     print(i)
#     print(img.shape)
#     num_pixel = img.shape[0]*img.shape[1]
#     print('num_pixel',num_pixel)
#     
#  
#     sum_pixel = np.sum(img,axis = 0)
#     print('sum_pixel0', sum_pixel.shape)
#     sum_pixel = np.sum(sum_pixel,axis = 0)
#     print('sum_pixel1', sum_pixel.shape)
#     print('sum_pixel1', sum_pixel )
#     
#     R = sum_pixel[0]
#     G = sum_pixel[1]
#     B = sum_pixel[2]
#     
#     print('RGB', R,G,B )
#     
#     print('img_sum0',img_sum)
#     img_sum += sum_pixel
#     print('img_sum1',img_sum)
#     total_pixel += num_pixel
#     
#     print('sum_RGB',sum_R,sum_G,sum_B )
#     sum_R += R
#     sum_G += G
#     sum_B += B
#     print('sum_RGB',sum_R,sum_G,sum_B )
#     i += 1
#     
#     '''
#     # if i ==2:
#     if i ==191:
#         break
#     '''
#     
# print('\n')
# print('img_sum',img_sum)
# print('total_pixel',total_pixel)
# mean = img_sum/total_pixel
# print('mean',mean)
# 
# mean_R = sum_R/total_pixel
# mean_G = sum_G/total_pixel
# mean_B= sum_B/total_pixel
# print('mean_RGB',mean_R,mean_G,mean_B )
# =============================================================================


# car 
# =============================================================================
# mat_anno = '/data/kb/tanyuanyong/TransFG-master/data/car/devkit/cars_train_annos.mat'
# data_dir = '/data/kb/tanyuanyong/TransFG-master/data/car/cars_train'
# =============================================================================
# =============================================================================
# mat_anno = '/data/kb/tanyuanyong/TransFG-master/data/car/devkit/cars_test_annos.mat'
# data_dir = '/data/kb/tanyuanyong/TransFG-master/data/car/cars_test'
# full_data_set = io.loadmat(mat_anno)
# car_annotations = full_data_set['annotations']
# car_annotations = car_annotations[0]
# 
# 
# i = 0
# img_sum = 0
# total_pixel = 0
# sum_R = 0
# sum_G = 0
# sum_B = 0
# for idx in range(full_data_set['annotations'].shape[1]):
#     img_name = os.path.join(data_dir, car_annotations[idx][-1][0])
#     image = Image.open(img_name).convert('RGB')
#     img = np.array(image)
#     print(i)
#     print(img.shape)
#     num_pixel = img.shape[0]*img.shape[1]
#     print('num_pixel',num_pixel)
#     
#  
#     sum_pixel = np.sum(img,axis = 0)
#     print('sum_pixel0', sum_pixel.shape)
#     sum_pixel = np.sum(sum_pixel,axis = 0)
#     print('sum_pixel1', sum_pixel.shape)
#     print('sum_pixel1', sum_pixel )
#     
#     R = sum_pixel[0]
#     G = sum_pixel[1]
#     B = sum_pixel[2]
#     
#     print('RGB', R,G,B )
#     
#     print('img_sum0',img_sum)
#     img_sum += sum_pixel
#     print('img_sum1',img_sum)
#     total_pixel += num_pixel
#     
#     print('sum_RGB',sum_R,sum_G,sum_B )
#     sum_R += R
#     sum_G += G
#     sum_B += B
#     print('sum_RGB',sum_R,sum_G,sum_B )
#     i += 1
#     
#     '''
#     # if i ==2:
#     if i ==191:
#         break
#     '''
#     
# print('\n')
# print('img_sum',img_sum)
# print('total_pixel',total_pixel)
# mean = img_sum/total_pixel
# print('mean',mean)
# 
# mean_R = sum_R/total_pixel
# mean_G = sum_G/total_pixel
# mean_B= sum_B/total_pixel
# print('mean_RGB',mean_R,mean_G,mean_B )
# =============================================================================


root = '/data/kb/tanyuanyong/TransFG-master/data/FGVC_aircraft/'

train_img_path = os.path.join(root, 'data', 'images')
test_img_path = os.path.join(root, 'data', 'images')
train_label_file = open(os.path.join(root, 'data', 'train.txt'))
test_label_file = open(os.path.join(root, 'data', 'test.txt'))
bbox_file = open(os.path.join(root, 'data', 'images_box.txt'))



train_img_pathlabel = []
train_img_name = []

test_img_pathlabel = []
test_img_name = []

train_bbox = []
test_bbox = []

for line in train_label_file:
    image_name = line.split('.')[0]
    train_img_name.append(image_name)
    train_img_pathlabel.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])


for line in test_label_file:
    image_name = line.split('.')[0]
    test_img_name.append(image_name)
    test_img_pathlabel.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])




ordered_train_img_pathlabel = []
ordered_test_img_pathlabel = []
for line in bbox_file:
    if line.split(' ')[0] in train_img_name :
        # print('line.split(' ')[0]',line.split(' ')[0])
        train_bbox.append(line.split('\n')[:-1][0])   # 按照image_box顺序  test_img_pathlabel顺序不对
        
        # print('index',train_img_name.index(line.split(' ')[0]))
        image_index = train_img_name.index(line.split(' ')[0])
        ordered_train_img_pathlabel.append(train_img_pathlabel[image_index])

    if line.split(' ')[0] in test_img_name :
        test_bbox.append(line.split('\n')[:-1][0])
        
        image_index = test_img_name.index(line.split(' ')[0])
        ordered_test_img_pathlabel.append(test_img_pathlabel[image_index])



i = 0
img_sum = 0
total_pixel = 0
sum_R = 0
sum_G = 0
sum_B = 0

count = len(ordered_test_img_pathlabel)

for idx in range(count):
    img, target = imageio.imread(ordered_test_img_pathlabel[idx][0]), ordered_test_img_pathlabel[idx][1]
    if len(img.shape) == 2:
        img = np.stack([img] * 3, 2)
    
    print(i)
    print(img.shape)
    num_pixel = img.shape[0]*img.shape[1]
    print('num_pixel',num_pixel)
    
 
    sum_pixel = np.sum(img,axis = 0)
    print('sum_pixel0', sum_pixel.shape)
    sum_pixel = np.sum(sum_pixel,axis = 0)
    print('sum_pixel1', sum_pixel.shape)
    print('sum_pixel1', sum_pixel )
    
    R = sum_pixel[0]
    G = sum_pixel[1]
    B = sum_pixel[2]
    
    print('RGB', R,G,B )
    
    print('img_sum0',img_sum)
    img_sum += sum_pixel
    print('img_sum1',img_sum)
    total_pixel += num_pixel
    
    print('sum_RGB',sum_R,sum_G,sum_B )
    sum_R += R
    sum_G += G
    sum_B += B
    print('sum_RGB',sum_R,sum_G,sum_B )
    i += 1
    
    '''
    # if i ==2:
    if i ==191:
        break
    '''
    
print('\n')
print('img_sum',img_sum)
print('total_pixel',total_pixel)
mean = img_sum/total_pixel
print('mean',mean)

mean_R = sum_R/total_pixel
mean_G = sum_G/total_pixel
mean_B= sum_B/total_pixel
print('mean_RGB',mean_R,mean_G,mean_B )



