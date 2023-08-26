import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor


class TrOCRDataset(Dataset):

    '''
        Subject - Text Recognition.
        This class is a wrapper around the dataset.
        It is used to provide a consistent way to iterate over datasets.
    '''

    def __init__(self, data: pd.DataFrame, processor: object = None, max_target_length: int = 150, 
                 overfit: bool = False, overfit_batch_size: int = 32,
                 processor_pretrained_path: str = "microsoft/trocr-base-handwritten") -> None:
        '''
            Initial definition for the OCRDataset class.
            Input params:
                data - a pandas DataFrame representing the dataset.
                processor - an object representing the TrOCRProcessor taken from Hugging Face.
                            it is a wrapper around the embedders.
                                ViTFeatureExtractor - used to resize and normalize the image.
                                RobertaTokenizer - used to tokenize the text | encode and decode the text.
                max_target_length - an integer representing the maximum length of the target.
                overfit - a boolean representing whether or not to overfit, i.e yielding same
                overfit_batch_size - an integer representing the batch size to use for overfitting.
                processor_pretrained_path - a string representing the path to the pretrained processor.
            Returns: None.
        '''
        super().__init__()
        self.data = data
        self.processor = processor if processor is not None else TrOCRProcessor.from_pretrained(processor_pretrained_path)
        self.max_target_length = max_target_length
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
            Returns: a dict containing image and its label.
        '''
        raw_instance = self.data.iloc[index]
        instance = {
            "pixel_values": Image.open(raw_instance.image_path).convert("RGB"),
            "labels": raw_instance.text
        }
        instance["pixel_values"] = self.processor(instance["pixel_values"], return_tensors="pt").pixel_values
        instance["labels"] = self.processor.tokenizer(instance["labels"], padding="max_length", max_length=self.max_target_length).input_ids
        instance["labels"] = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in instance["labels"]
        ]
        if len(instance["labels"]) > self.max_target_length:
            print(f"Clipping the labels from {len(instance['labels'])} to {self.max_target_length}")
            instance["labels"] = instance["labels"][:self.max_target_length]
        return {
            "pixel_values": instance["pixel_values"].squeeze(), 
            "labels": torch.tensor(instance["labels"])
        }


class TrOCRData:

    def __init__(self, 
                 train_df: pd.DataFrame = None,
                 val_df: pd.DataFrame = None,
                 test_df: pd.DataFrame = None,
                 train_batch_size: int = 32,
                 val_batch_size: int = 32,
                 test_batch_size: int = 32,
                 train_shuffle: bool = True,
                 val_shuffle: bool = False,
                 test_shuffle: bool = False,
                 processor: object = None,
                 processor_pretrained_path: str = "microsoft/trocr-base-handwritten",
                 max_target_length: int = 150, 
                 overfit: bool = False, 
                 overfit_batch_size: int = 32,
                 verbose: bool = False,
                 num_workers: int = None,
                 pin_memory: bool = False,
                 ) -> None:
        '''
            Initial definition for the OCRData class.
            Input params:
                train_df - a pandas DataFrame representing the training dataset.
                val_df - a pandas DataFrame representing the validation dataset.
                test_df - a pandas DataFrame representing the testing dataset.
                train_batch_size - an integer representing the batch size for training.
                val_batch_size - an integer representing the batch size for validation.
                test_batch_size - an integer representing the batch size for testing.
                train_shuffle - a boolean representing whether or not to shuffle the training data.
                val_shuffle - a boolean representing whether or not to shuffle the validation data.
                test_shuffle - a boolean representing whether or not to shuffle the testing data.
                processor - an object representing the TrOCRProcessor taken from Hugging Face.
                            it is a wrapper around the embedders.
                                ViTFeatureExtractor - used to resize and normalize the image.
                                RobertaTokenizer - used to tokenize the text | encode and decode the text.
                processor_pretrained_path - a string representing the path to the pretrained processor.
                max_target_length - an integer representing the maximum length of the target.
                overfit - a boolean representing whether or not to overfit.
                overfit_batch_size - an integer representing the batch size to use for overfitting.
                verbose - a boolean representing whether or not to print the dataset.
                num_workers - an integer representing the number of workers to use for loading the data.
                pin_memory - a boolean representing whether or not to pin the memory.
            Returns: None.
        '''
        self.train_df = train_df.loc[::-1]
        self.val_df = val_df
        self.test_df = test_df
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle
        self.processor = processor if processor is not None else TrOCRProcessor.from_pretrained(processor_pretrained_path)
        self.max_target_length = max_target_length
        self.overfit = overfit
        self.overfit_batch_size = overfit_batch_size
        self.verbose = verbose
        self.pin_memory = pin_memory
        self.num_workers = os.cpu_count() // 2 if num_workers is None else num_workers

    def get_data_loaders(self) -> tuple:
        '''
            Returns the data loaders.
            Input params: None.
            Returns: a tuple containing the training, validation and testing data loaders.
        '''
        train_dataset = TrOCRDataset(self.train_df, self.processor, self.max_target_length, self.overfit, self.overfit_batch_size)
        val_dataset = TrOCRDataset(self.val_df, self.processor, self.max_target_length, self.overfit, self.overfit_batch_size)
        test_dataset = TrOCRDataset(self.test_df, self.processor, self.max_target_length, self.overfit, self.overfit_batch_size)
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=self.train_shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=self.val_batch_size, shuffle=self.val_shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=self.test_shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)
        return (train_loader, val_loader, test_loader)

    def visualize(self) -> None:
        '''
            This function is used to visualize the dataset results.
            Input params: None.
            Returns: None.
        '''
        train_loader, _, _ = self.get_data_loaders()
        encoding = next(iter(train_loader))
        print(f"Before squeezing the image size -> {encoding['pixel_values'].shape}")
        print(f"Before squeezing the label size -> {encoding['labels'].shape}")
        image, label = encoding["pixel_values"][0].squeeze(), encoding["labels"][0].squeeze()
        print(f"The input image size -> {image.shape}")
        print(f"The label size -> {label.shape}")
        label[label == -100] = self.processor.tokenizer.pad_token_id
        label_decoded = self.processor.tokenizer.decode(label, skip_special_tokens=True)
        print(f"The text in the image -> {label_decoded}")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image.permute(1, 2, 0))
        ax.set_aspect(aspect=0.25)