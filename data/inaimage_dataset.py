import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from PIL import Image
import os
import numpy as np
from PIL import Image, ImageDraw
import math
import json
import cv2

def RandomBrush(
    max_tries,
    s,
    min_num_vertex = 4,
    max_num_vertex = 18,
    mean_angle = 2*math.pi / 5,
    angle_range = 2*math.pi / 15,
    min_width = 12,
    max_width = 48):
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask

def RandomMask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)
        MultiFill(int(10 * coef), s // 2)
        MultiFill(int(5 * coef), s)
        mask = np.logical_and(mask, 1 - RandomBrush(int(20 * coef), s))
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)


class InaImageDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--output_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--min_hole', type=float, required=True,
                            help='value to the hole min')
        parser.add_argument('--max_hole', type=float, required=True,
                            help='value to the hole max')
        parser.add_argument('--dataset_state', type=str, required=False, default='train',
                            help='value to the hole max')

        return parser
    
    def initialize(self, opt):
        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)

        self.root = opt.image_dir
        self.dst_root = opt.output_dir
        self.mask_generate = RandomMask
        self.min_hole = opt.min_hole
        self.max_hole = opt.max_hole
        self.state = opt.dataset_state

        transform_list = [
                transforms.Resize((opt.crop_size, opt.crop_size), 
                    interpolation=Image.NEAREST),
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                ]
        self.image_transform = transforms.Compose(transform_list)
        self.mask_transform = transforms.Compose([
            transforms.Resize((opt.crop_size, opt.crop_size),interpolation=Image.NEAREST),
            transforms.ToTensor()
            ])

        if self.state:
            file_annotation = self.root + '/train.json'

        else:
            file_annotation = self.root + '/val.json'

        fp = open(file_annotation, 'r')
        jsontext = json.load(fp)
        # from json get imagepath and class
        imgidpath_in = {i['id']: self.root + '/' + i['file_name'] for i in jsontext['images']}
        imgidpath_out = {i['id']: self.dst_root + '/' + i['file_name'] for i in jsontext['images']}
        #   info of class
        imgidclass = {i['image_id']: i['category_id'] for i in jsontext['annotations']}
        # get sample with path  in and out
        samples = [(imgidpath_in[i], imgidpath_out[i]) for i in imgidpath_in.keys()]
        # i dont know
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)


    def scale_image(self,image):
        
        width, height = image.size
        size = min(width, height)

        # 计算裁剪的左上角坐标
        left = (width - size) // 2
        top = (height - size) // 2

        # 计算裁剪的右下角坐标
        right = left + size
        bottom = top + size

        # 从中间裁剪
        cropped_image = image.crop((left, top, right, bottom))

        return cropped_image,size,(left, top, right, bottom)

    def __getitem__(self, index):
        # input image (real images)
        image_path, output_path = self.samples[index]
        # output_path = self.output_paths[index]
        # image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image,size,rect = self.scale_image(image)
        # image_tensor = self.image_transform(image)
        mask = self.mask_generate(s=size, hole_range=[self.min_hole,self.max_hole])
        mask_2d = np.squeeze(mask)
        mask_uint8 = (255 * mask_2d).astype(np.uint8)
        mask = Image.fromarray(mask_uint8)
        mask = mask.convert("L")
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor>0).float()
        input_dict = {
                    #   'image': None,
                      'mask': mask_tensor,
                      'output_path': output_path,
                      'image_path': image_path,
                      'size': size,
                      'rect':rect
                      }
        return input_dict
    
# if __name__ == '__main__':
#     dataset = InaImageDataset("/data/kb/tanyuanyong/TransFG-master/data/tmp/INaturalist2021","./INaturalist2021_MASK_GEN")