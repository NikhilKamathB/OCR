import torch
import numpy as np
from PIL import Image
from utils import normalize_image, generate_region_affinity_heatmap


class Resize:

    '''
        Resize the image to a given size.
    '''

    def __init__(self, size: tuple = (512, 512)) -> None:
        '''
            Initial definition for the Resize class.
            Input params:
                size - a tuple representing the size (height, width) 
                       to resize the image to.
            Returns: None.
        '''
        assert isinstance(size, tuple), "`size` must be a tuple."
        assert isinstance(size[0], int) and isinstance(size[1], int), "`size` must be a tuple of integers."
        self.size = size
    
    def __call__(self, instance: dict) -> dict:
        '''
            Input params: instance - a dictionary representing a data instance.
            Returns: a dictionary representing a data instance with image resized.
        '''
        new_height, new_width = self.size
        instance["image"] = instance["image"].resize((new_width, new_height))
        instance["region_heatmap"], instance["affinity_heatmap"] = \
            self._get_region_and_affinity_heatmap(
                image_annotations_path=instance["annotations"], image=instance["image"]
            )
        instance["image"] = np.array(instance["image"])
        return instance

    def _get_region_and_affinity_heatmap(self, image_annotations_path:str, image: Image) -> np.ndarray:
        '''
            Get region and affinity heatmap.
            Input params:
                annotation_path: path to annotation file.

            Returns: a np.ndarray representing the region and affinity heatmap.
        '''
        return generate_region_affinity_heatmap(image_annotations_path=image_annotations_path, image=image)


class Normalize:

    '''
        Normalize the image.
    '''

    def __call__(self, instance: dict) -> dict:
        '''
            Input params: instance - a dictionary representing a data instance.
            Returns: a dictionary representing a data instance with image normalized.
        '''
        instance["image"] = normalize_image(image=instance["image"])
        return instance


class ToTensor:

    '''
        Convert data to tensor.
    '''

    def __call__(self, instance: dict) -> dict:
        '''
            Input params: instance - a dictionary representing a data instance.
            Returns: a dictionary representing a data instance with data converted to tensor.
        '''
        instance["image"] = torch.from_numpy(np.array(instance["image"]))
        instance["region_heatmap"] = torch.from_numpy(instance["region_heatmap"])
        instance["affinity_heatmap"] = torch.from_numpy(instance["affinity_heatmap"])
        return instance