# https://medium.com/analytics-vidhya/image-text-recognition-738a368368f5

import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms
from .transforms import *
from .config import config as conf
from .utils import *


class CustomDataset(Dataset):
  
    def __init__(self, data, transforms=None, label_convertor=None, overfit=False, overfit_batch_size=32):
        self.desired_image_width = conf.TRANSFORM_SIZE_WIDTH
        self.desired_image_height = conf.TRANSFORM_SIZE_HEIGHT
        self.max_length = conf.TEXT_MAX_LENGTH
        self.data = data
        self.transforms = transforms
        self.label_converter = label_convertor
        self.overfit = overfit
        self.overfit_batch_size = overfit_batch_size
  
    def __len__(self):
        return len(self.data) if not self.overfit else self.overfit_batch_size

    def __getitem__(self, index):
        image = Image.open(self.data.iloc[index]['image_path_nbs'])
        label = self.data.iloc[index]['label']
        processed_label, processed_label_length = self.label_converter.encode(text=label, max_length=self.max_length)
        sample = {
            'image': np.array(image),
            'label': processed_label,
            'label_length': processed_label_length
        }
        sample_output = self.transforms(sample)
        return sample_output['image'], sample_output['label'], sample_output['label_length']


class Data:

    def __init__(self, data_train=None, data_val=None, data_test=None, transforms=None, train_batch_size=None, val_batch_size=None, test_batch_size=None, label_convertor=None, shuffle=False, overfit=False, overfit_batch_size=32):
        self.resize_width = conf.TRANSFORM_SIZE_WIDTH
        self.resize_height = conf.TRANSFORM_SIZE_HEIGHT
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.label_convertor = label_convertor
        self.shuffle = shuffle
        # self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.train_batch_size, self.val_batch_size, self.test_batch_size = 32 if train_batch_size is None else train_batch_size, 32 if train_batch_size is None else val_batch_size, 32 if test_batch_size is None else test_batch_size
        self.overfit = overfit
        self.overfit_batch_size = overfit_batch_size
        self.transforms = self.get_transforms() if transforms is None or not isinstance(transforms, dict) else transforms

    def get_transforms(self):
        TRANSFORMS = {
            'train': transforms.Compose([
                Resize(size=(self.resize_width, self.resize_height)),
                ToTensor(),
                Normalize(),
            ]),
            'validation': transforms.Compose([
                Resize(size=(self.resize_width, self.resize_height)),
                ToTensor(),
                Normalize()
            ]),
            'test': transforms.Compose([
                Resize(size=(self.resize_width, self.resize_height)),
                ToTensor(),
                Normalize()
            ])
        }
        return TRANSFORMS

    def get_loaders(self):
        train_loader = torch.utils.data.DataLoader(CustomDataset(data=self.data_train, transforms=self.transforms['train'], label_convertor=self.label_convertor, overfit=self.overfit, overfit_batch_size=self.overfit_batch_size), batch_size=self.train_batch_size, shuffle=self.shuffle)
        val_loader = torch.utils.data.DataLoader(CustomDataset(data=self.data_val, transforms=self.transforms['validation'], label_convertor=self.label_convertor, overfit=self.overfit, overfit_batch_size=self.overfit_batch_size), batch_size=self.val_batch_size, shuffle=self.shuffle)
        test_loader = torch.utils.data.DataLoader(CustomDataset(data=self.data_test, transforms=self.transforms['test'], label_convertor=self.label_convertor, overfit=self.overfit, overfit_batch_size=self.overfit_batch_size), batch_size=self.test_batch_size, shuffle=self.shuffle)
        return train_loader, val_loader, test_loader

    def visualize_loaders(self):
        trainloader, _, _ = self.get_loaders()
        dataiter = iter(trainloader)
        images, labels, _ = dataiter.next()
        visualize(data=images, labels=labels, label_convertor=self.label_convertor)