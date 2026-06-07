import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import json
from PIL import Image
from tqdm import tqdm
from absl import flags
import logging
from io import BytesIO
from utils import matrix_to_6d_rotation

FLAGS = flags.FLAGS


class YCBVDataset(Dataset):
    def __init__(self, data_dir='./data/ycbv/v1/', split='train', transform=None, 
                 preload_images=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.data = json.load(open(os.path.join(self.data_dir, 
                                                f'./ycbv_{split}.json')))
        for k, v in self.data.items():
            v['key_name'] = k
        self.data = list(self.data.values())
        self.num_classes = 13
        self.preload_images = preload_images
        if self.preload_images:
            self._preload_images()

    def _preload_images(self,):
        logging.info(f'Preloading {self.split} Images Strings into memory')
        self.image_strings = {}
        for i in range(len(self.data)):
            img_path = self._get_image_path(i)
            if img_path not in self.image_strings.keys():
                image_string = open(img_path, 'rb').read()
                self.image_strings[img_path] = image_string
        logging.info(f'Finished Preloading {self.split} Images.')
    
    def __len__(self):
        return len(self.data)
    
    def _get_image(self, idx):
        if self.preload_images:
            image_string = self.image_strings[self._get_image_path(idx)]
        else:
            img_path = self._get_image_path(idx)
            image_string = open(img_path, 'rb').read()
        img = Image.open(BytesIO(image_string)).convert('RGB') 
        return img
    
    def _get_image_path(self, idx):
        img_name = self.data[idx]['img_name']
        return os.path.join(self.data_dir, 'rgb', img_name)

    def __getitem__(self, idx):
        item = self.data[idx].copy()
        obj_class = item['obj_id']
        R = torch.tensor(item['cam_R_m2c'], dtype=torch.float32).reshape(3, 3)
        # rot_6d = matrix_to_6d_rotation(R)
        t = torch.tensor(item['cam_t_m2c'], dtype=torch.float32).reshape(3, 1) / 1000.
        img = self._get_image(idx)
        bbox = torch.tensor(item['bbox_visib'], dtype=torch.float32).reshape(4)
        bbox_buffer = bbox.numpy()
        bbox_data = bbox_buffer.copy()
        if self.transform:
            img = img.crop((bbox_data[0], bbox_data[1], bbox_data[0] + bbox_data[2], bbox_data[1] + bbox_data[3]))
            img = img.resize((224, 224))
            img = self.transform(img)
        return img, bbox, obj_class, R, t, item['key_name'] # Previously did return R


