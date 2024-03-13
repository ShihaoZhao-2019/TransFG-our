import os
import json
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from autoaugment import AutoAugImageNetPolicy
import time
class INat2021(data.Dataset):
    def __init__(
            self,
            root,
            transform=None,
            train=False):
        images, class_to_idx,images_info = self.find_images_and_targets(root,train)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. ')
        self.root = root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.images_info = images_info
        self.transform = transform
        

    def find_images_and_targets(self,root,istrain=False):
        if os.path.exists(os.path.join(root,'train.json')):
            with open(os.path.join(root,'train.json'),'r') as f:
                train_class_info = json.load(f)
        else:
            raise ValueError(f'not eixst file {root}/train.json or {root}/train_mini.json')
        with open(os.path.join(root,'val.json'),'r') as f:
            val_class_info = json.load(f)
        categories_2021 = [x['name'].strip().lower() for x in val_class_info['categories']]
        class_to_idx = {c: idx for idx, c in enumerate(categories_2021)}
        id2label = dict()
        for categorie in train_class_info['categories']:
            id2label[int(categorie['id'])] = categorie['name'].strip().lower()
        class_info = train_class_info if istrain else val_class_info
        
        images_and_targets = []
        images_info = []

        for image,annotation in zip(class_info['images'],class_info['annotations']):
            file_path = os.path.join(root,image['file_name'])
            id_name = id2label[int(annotation['category_id'])]
            target = class_to_idx[id_name]
            date = image['date']
            latitude = image['latitude']
            longitude = image['longitude']
            location_uncertainty = image['location_uncertainty']
            images_info.append({'date':date,
                    'latitude':latitude,
                    'longitude':longitude,
                    'location_uncertainty':location_uncertainty,
                    'target':target}) 
            images_and_targets.append((file_path,target))
        return images_and_targets,class_to_idx,images_info

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            return img, target
        else:
            raise ValueError(f'please set a transform')
        
    def __len__(self):
        return len(self.samples)
   
   
   
   
train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                            transforms.RandomCrop((224, 224)),
                            transforms.RandomHorizontalFlip(),
                            AutoAugImageNetPolicy(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                            transforms.CenterCrop((448, 448)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dataset = INat2021(root = '/data/kb/tanyuanyong/TransFG-master/data/INat2021',transform = train_transform,train = True)


from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
train_sampler = RandomSampler(dataset)
train_loader = DataLoader(dataset,
                            sampler=train_sampler,
                            batch_size=32,
                            num_workers=4,
                            drop_last=True,
                            pin_memory=True)

current_time = time.time()

for i in train_loader:
    lasttime = current_time
    current_time = time.time()
    diff_time = current_time - lasttime
    print("run time:", diff_time, "s")