import cv2
import copy
import torch
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F


# Resize
class Resize:

    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        self.size = size
    
    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if isinstance(self.size, int):
            if h > w:
                new_h, new_w = self.size * h / w, self.size
            else:
                new_h, new_w = self.size, self.size * w / h
        else:
            new_w, new_h = self.size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))
        return {'image': img.astype(np.float32), 'label': sample['label'], 'label_length': sample['label_length']}


# Convert to tensor.
class ToTensor:

    def __call__(self, sample):
        image, label, label_length = sample['image'], sample['label'], sample['label_length']
        if(len(image.shape) == 2):
            image = image.reshape(image.shape[0], image.shape[1], 1)
        # H x W x C  ->  C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        return {'image': image.float(), 'label':torch.IntTensor(label), 'label_length': torch.IntTensor(label_length)}


# Normalize.
class Normalize:

    def __call__(self, sample):
        image = sample['image']
        image_copy = copy.deepcopy(image)
        image_copy = image_copy/image_copy.max()
        return {'image': image_copy, 'label': sample['label'], 'label_length': sample['label_length']}