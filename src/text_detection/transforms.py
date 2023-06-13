import torch
import numpy as np
from PIL import Image
from .utils import normalize_image, generate_region_affinity_heatmap


class Resize:

    '''
        Resize the image to a given size.
    '''

    def __init__(self, size: tuple = (512, 512), heatmap_size: tuple = (512, 512), fill_method: str = "gaussain") -> None:
        '''
            Initial definition for the Resize class.
            Input params:
                size - a tuple representing the size (height, width) 
                       to resize the image to.
                heatmap_size - a tuple representing the size (height, width) for
                                the heatmap.
                fill_method - a string representing the method to use to fill the heatmap.
                              default - "gaussian"
            Returns: None.
        '''
        assert isinstance(size, tuple), "`size` must be a tuple."
        assert isinstance(size[0], int) and isinstance(size[1], int), "`size` must be a tuple of integers."
        assert isinstance(heatmap_size, tuple), "`heatmap_size` must be a tuple."
        assert isinstance(heatmap_size[0], int) and isinstance(heatmap_size[1], int), "`heatmap_size` must be a tuple of integers." 
        self.size = size
        self.heatmap_size = heatmap_size
        self.fill_method = fill_method
    
    def __call__(self, instance: dict) -> dict:
        '''
            Input params: instance - a dictionary representing a data instance.
            Returns: a dictionary representing a data instance with image resized.
        '''
        new_height, new_width = self.size
        instance["image"] = np.asarray(instance["image"].resize((new_width, new_height)))
        instance["region_heatmap"], instance["affinity_heatmap"] = \
            self._get_region_and_affinity_heatmap(
                image_annotations_path=instance["annotations"], image=instance["image"]
            )
        return instance

    def _get_region_and_affinity_heatmap(self, image_annotations_path:str, image: Image) -> np.ndarray:
        '''
            Get region and affinity heatmap.
            Input params:
                annotation_path: path to annotation file.

            Returns: a np.ndarray representing the region and affinity heatmap.
        '''
        return generate_region_affinity_heatmap(image_annotations_path=image_annotations_path, image=image, heatmap_size=self.heatmap_size, fill_method=self.fill_method)


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
        instance["image"] = torch.permute(torch.from_numpy(np.array(instance["image"])).type(torch.float32), (2, 0, 1))
        instance["region_heatmap"] = torch.from_numpy(instance["region_heatmap"]).type(torch.float32)
        instance["affinity_heatmap"] = torch.from_numpy(instance["affinity_heatmap"]).type(torch.float32)
        return instance