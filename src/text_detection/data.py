import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from .transforms import *


class OCRDataset(Dataset):

    '''
        This class is a wrapper around the dataset.
        It is used to provide a consistent way to iterate over datasets.
    '''

    def __init__(self, data: pd.DataFrame, transforms: dict,
                 overfit: bool, overfit_batch_size: int = 32) -> None:
        '''
            Initial definition for the OCRDataset class.
            Input params: 
                data - a pandas DataFrame representing the dataset.
                transforms - a dictionary of transforms to be applied to the dataset.
                overfit - a boolean representing whether or not to overfit, i.e yielding same
                          set of instances.
                overfit_batch_size - an integer representing the batch size to use for overfitting.
            Returns: None.
        '''
        super().__init__()
        self.data = data
        self.transforms = transforms
        self.overfit = overfit
        self.overfit_batch_size = overfit_batch_size
    
    def __len__(self) -> int:
        '''
            Input params: None.
            Returns: an integer representing the length of the dataset.
        '''
        if self.overfit:
            return self.overfit_batch_size
        return len(self.data)

    def __getitem__(self, index) -> tuple:
        '''
            Input params: index - an integer representing the index of the instance to retrieve.
            Returns: a tuple representing image, region and affitnity heatmap.
        '''
        raw_instance = self.data.iloc[index]
        instance = {
            "image": Image.open(raw_instance.image).convert("RGB"),
            "annotations": raw_instance.annotations,
            "region_heatmap": None,
            "affinity_heatmap": None
        }
        output_instance = self.transforms(instance)
        return (
            output_instance["image"], 
            output_instance["region_heatmap"], 
            output_instance["affinity_heatmap"]
        )


class OCRData:

    def __init__(self, 
                train_data: pd.DataFrame = None,
                val_data: pd.DataFrame = None,
                test_data: pd.DataFrame = None,
                resize: tuple = (768, 768),
                heatmap_size: tuple = (384, 384),
                train_batch_size: int = 32,
                val_batch_size: int = 32,
                test_batch_size: int = 32,
                train_shuffle: bool = True,
                val_shuffle: bool = False,
                test_shuffle: bool = False,
                transforms: dict = None,
                overfit: bool = False,
                overfit_batch_size: int = 32,
                num_workers: int = None,
                pin_memory: bool = False,
                fill_method: str = "gaussian",
                ) -> None:
        '''
            Initial definition for the OCRData class.
            Input params:
                train_data - a pandas DataFrame representing the training data.
                val_data - a pandas DataFrame representing the validation data.
                test_data - a pandas DataFrame representing the test data.
                resize - a tuple representing the size (height, width).
                heatmap_size - a tuple representing the size (height, width) of the heatmap.
                train_batch_size - an integer representing the batch size to use for training.
                val_batch_size - an integer representing the batch size to use for validation.
                test_batch_size - an integer representing the batch size to use for testing.
                train_shuffle - a boolean representing whether or not to shuffle the training data.
                val_shuffle - a boolean representing whether or not to shuffle the validation data.
                test_shuffle - a boolean representing whether or not to shuffle the test data.
                transforms - a dictionary of transforms to be applied to the dataset - train, val and test.
                overfit - a boolean representing whether or not to overfit, i.e yielding same set of instances.
                overfit_batch_size - an integer representing the batch size to use for overfitting.
                num_workers - an integer representing the number of workers to use for loading data.
                pin_memory - a boolean representing whether or not to pin memory.
                fill_method - a string representing the method to use for filling the heatmap, default is gaussian.
        '''
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.resize = resize
        self.heatmap_size = heatmap_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle
        self.overfit = overfit
        self.overfit_batch_size = overfit_batch_size
        self.pin_memory = pin_memory
        self.fill_method = fill_method
        self.num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
        self.transforms = self.get_tranforms() if transforms is None else transforms
    
    def get_tranforms(self) -> dict:
        '''
            Define your transforms here.
            Input params: None.
            Returns: a dictionary of transforms to be applied to the dataset - train, val and test.
        '''
        return {
            "train": T.Compose([
                Resize(size=self.resize, heatmap_size=self.heatmap_size, fill_method=self.fill_method),
                Normalize(),
                ToTensor()
            ]),
            "validate": T.Compose([
                Resize(size=self.resize, heatmap_size=self.heatmap_size, fill_method=self.fill_method),
                Normalize(),
                ToTensor()
            ]),
            "test": T.Compose([
                Resize(size=self.resize, heatmap_size=self.heatmap_size, fill_method=self.fill_method),
                Normalize(),
                ToTensor()
            ])
        }

    def get_data_loaders(self) -> tuple:
        '''
            Input params: None.
            Returns: a tuple of data loaders for train, val and test sets.
        '''
        train_dataset = OCRDataset(
            data=self.train_data,
            transforms=self.transforms["train"],
            overfit=self.overfit,
            overfit_batch_size=self.overfit_batch_size
        )
        val_dataset = OCRDataset(
            data=self.val_data,
            transforms=self.transforms["validate"],
            overfit=self.overfit,
            overfit_batch_size=self.overfit_batch_size
        )
        test_dataset = OCRDataset(
            data=self.test_data,
            transforms=self.transforms["test"],
            overfit=self.overfit,
            overfit_batch_size=self.overfit_batch_size
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.test_batch_size,
            shuffle=self.test_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        return (train_loader, val_loader, test_loader)

    def visualize(self, number_of_subplots: int = 8, columns: int = 4, rows: int = 2, figsize=(30, 10), alpha: float = 0.5) -> None:
        '''
            Visualize the output of the loader
            Input params:
                number_of_subplots - an integer representing the number of subplots to plot.
                columns - an integer representing the number of columns to use for plotting.
                rows - an integer representing the number of rows to use for plotting.
                figsize - a tuple representing the size of the figure.
                alpha - a float representing the alpha value for the heatmap.
            Returns: None.
        '''
        assert number_of_subplots == columns * rows, "`number_of_subplots` must be equal to the product `columns` and `rows` for plotting convenience."
        train_loader, _, _ = self.get_data_loaders()
        _, ax = plt.subplots(rows, columns, figsize=figsize)
        image, region_heatmap, affinity_heatmap = next(iter(train_loader))
        image_idx = 0
        for r in range(rows):
            for c in range(0, columns, 2):
                bg_image = image[image_idx].permute(1, 2, 0).numpy()
                bg_image = cv2.resize(bg_image, (self.heatmap_size[1], self.heatmap_size[0]))
                ax[r, c].imshow(bg_image)
                ax[r, c].imshow(region_heatmap[image_idx], alpha=alpha)
                ax[r, c].set_title(f"Image {image_idx} | Region Heatmap")
                ax[r, c+1].imshow(bg_image)
                ax[r, c+1].imshow(affinity_heatmap[image_idx], alpha=alpha)
                ax[r, c+1].set_title(f"Image {image_idx} | Affinity Heatmap")
                image_idx += 1
