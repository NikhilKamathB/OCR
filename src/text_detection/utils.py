import cv2
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm


def str2bool(v):
    '''
        Convert string to boolean, basically used by the cmd parser.
    '''
    return v.lower() in ("yes", "Yes", "YES", "y", "true", "True", "TRUE", "t", "1")

def get_annotated_file(image_path: str) -> str:
    '''
        Get annotated file path from image path.
        Input params:
            image_path: path to image file.
        Returns: annotated file path.
    '''
    assert image_path is not None, 'Image path is None. Provide a valid image path.'
    return image_path.replace('.jpg', '_ocr.json')

def split_data(df: pd.DataFrame, split_ratio: list = [0.6, 0.2]) -> tuple:
    '''
        Split data into train, validation and test sets.
        Input params:
            df: pandas dataframe.
            split_ratio: list of split ratios. Total items in this list can be max 2, one corresponding
                        to train split and the other to validation split. Test split would
                        be the remaining. Sum of items in `split_ratio` must be less than 1.0.
        Returns: tuple of dataframes - (train, val) if length of `split_ratio` is 1 and (train, val, test)
                if the length of `split_ratio` is 2.
    '''
    assert len(split_ratio) <= 2, 'Length of `split_ratio` must be less than or equal to 2.'
    assert sum(split_ratio) < 1.0, 'Sum of items in `split_ratio` must be less than 1.0.'
    df = df.sample(frac=1, random_state=42)
    train_split_ratio = split_ratio[0]
    if len(split_ratio) == 2:
        val_split_ratio = split_ratio[1]
    else:
        val_split_ratio = None
    train_df = df.iloc[:int(len(df)*train_split_ratio)]
    if val_split_ratio is None:
        val_df = df.iloc[int(len(df)*train_split_ratio): ]
        return (train_df, val_df)
    val_df = df.iloc[int(len(df)*train_split_ratio): int(len(df)*(train_split_ratio + val_split_ratio))]
    test_df = df.iloc[int(len(df)*(train_split_ratio + val_split_ratio)): ]
    return (train_df, val_df, test_df)

def get_df(data_dir: str) -> pd.DataFrame:
    '''
        Get dataframe representing the dataset.
        Input params:
            data_dir: path to data directory.
        Returns: dataframe.
    '''
    assert data_dir is not None, 'Data directory is None. Provide a valid data directory.'
    data = glob.glob(f"{data_dir}/*.jpg")
    df = {"image": [], "annotations": []}
    for _, item in enumerate(data):
        df['image'].append(item)
        df['annotations'].append(get_annotated_file(item))
    return pd.DataFrame(df)

def normalize_image(image: np.ndarray, mean: tuple = (0.485, 0.456, 0.406),
                    variance: tuple = (0.229, 0.224, 0.225)) -> np.ndarray:
    '''
        Normalize image.
        Input params:
            image: np.ndarray of shape (image_height, image_width, channels[optional]).
            mean: mean of image.
            variance: variance of image.
        Returns: normalized image.
    '''
    image = image - np.array([mean[0]*255, mean[1]*255, mean[2]*255], dtype=np.float32)
    image = image / np.array([variance[0]*255, variance[1]*255, variance[2]*255], dtype=np.float32)
    image = np.clip(image, 0, 1).astype(np.float32)
    return image

def denormalize_image(image: np.ndarray, mean: tuple = (0.485, 0.456, 0.406),
                      variance: tuple = (0.229, 0.224, 0.225)) -> np.ndarray:
    '''
        Denormalize image.
        Input params:
            image: np.ndarray of shape (image_height, image_width, channels[optional]).
            mean: mean of image.
            variance: variance of image.
        Returns: denormalized image.
    '''
    image = image * np.array(variance)
    image = image + np.array(mean)
    image = image * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def get_gaussian_image(image: np.ndarray, mean: float = 0.0, stddev: float = 0.5, epsilon: float = 1e-6) -> np.ndarray:
    '''
        Get gaussian image.
        Input params:
            image: np.ndarray of shape (image_height, image_width, channels[optional]).
            mean: mean of gaussian distribution.
            stddev: standard deviation of gaussian distribution.
            epsilon: epsilon value to avoid division by zero.
        Returns: gaussian image.
    '''
    h, w = image.shape[: 2]
    if h == 0 or w == 0:
        return image
    x = np.linspace(-1, 1, image.shape[1])
    y = np.linspace(-1, 1, image.shape[0])
    x, y = np.meshgrid(x, y)
    image = norm.pdf(np.sqrt(x**2 + y**2), mean, stddev)
    image = (image - np.min(image)) / max((np.max(image) - np.min(image)), epsilon)
    return image

def get_solid_image(image: np.ndarray) -> np.ndarray:
    '''
        Get solid image.
        Input params:
            image: np.ndarray of shape (image_height, image_width, channels[optional]).
        Returns: solid image.
    '''
    h, w = image.shape[: 2]
    if h == 0 or w == 0:
        return image
    image = np.ones((h, w))
    return image

def visualize_image(image_path: str, annotation_file_path: str = None, 
                    annotation_color: tuple = (255, 0, 0)) -> None:
    '''
        Visualize image with annotations.
        Input params:
            image_path: path to image file
            annotation_file_path: path to annotation file
            annotation_color: color of annotation
        Returns: None
    '''
    assert image_path is not None, 'Image path is None. Provide a valid image path.'
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    if annotation_file_path is not None:
        with open(annotation_file_path, 'r') as file:
            annotations = json.load(file)
        for ind, annotation in enumerate(annotations['annotations']):
            if ind == 0:
                '''
                    First annotation represents the entire text as returned by
                    the google OCR parser, therefore skip it.
                '''
                continue
            image = cv2.rectangle(image, (annotation['x1'], annotation['y1']), 
                                  (annotation['x3'], annotation['y3']), 
                                  annotation_color, 2)
    plt.figure(figsize=(13, 13))
    plt.imshow(image)


def visualize_ndarray_image(images: list, opacity: list) -> None:
    '''
        Visualize ndarray image.
        Input params:
            images: list of images where each item is of the type np.ndarray.
            opacity: list of opacity values for each image listed above.
        Returns: None
    '''
    assert len(images) == len(opacity), 'Number of images `images.shape[0]` and length of `opacity` should be equal.'
    _, ax = plt.subplots(figsize=(13, 13))
    for ind, image in enumerate(images):
        ax.imshow(image, alpha=opacity[ind])
    plt.show()


def generate_region_affinity_heatmap(image_path: str = None, image_annotations_path: str = None, 
                                    image: np.ndarray = None, heatmap_size: tuple = None,
                                    fill_method: str = "gaussain") -> np.ndarray:
    '''
        Generate region heatmap for image.
        Input params:
            image_path: path to image file.
            image_annotations_path: path to annotation file.
            image: optional image of type np.ndarray.
            heatmap_size: size of heatmap (height, width).
            fill_method: method to fill the heatmap. Possible values are `gaussian` and `solid`.
                        default: `gaussian`.
        Returns: np.ndarray of shape (2, image_height, image_width).
    '''
    assert image_path is not None or image is not None, 'Either `image_path` or `image` should be provided.'
    assert (image_path is not None and image_annotations_path is None) or \
            (image_path is None and image_annotations_path is not None), 'Either `image_path` or `image_annotations_path` should be provided.'
    if image_annotations_path is None:
        image_annotations_path = get_annotated_file(image_path)
    if image is None:
        image = np.asarray(Image.open(image_path))
    if heatmap_size is None:
        image_height, image_width, _ = image.shape
    else:
        image_height, image_width = heatmap_size
    region_heatmap = np.zeros((image_height, image_width))
    affinity_heatmap = np.zeros((image_height, image_width))
    with open(image_annotations_path, 'r') as file:
        annotations = json.load(file)
    for ind, annotation in enumerate(annotations['annotations']):
        if ind == 0:
            '''
                First annotation represents the entire text as returned by
                the google OCR parser, therefore skip it.
            '''
            continue
        h1, h2 = min(int(annotation['y1_normalized']*image_height), int(annotation['y2_normalized']*image_height)), \
            max(int(annotation['y3_normalized']*image_height), int(annotation['y4_normalized']*image_height))
        w1, w2 = min(int(annotation['x1_normalized']*image_width), int(annotation['x4_normalized']*image_width)), \
            max(int(annotation['x2_normalized']*image_width), int(annotation['x3_normalized']*image_width))
        w1, h1 = max(w1, 0), max(h1, 0)
        w2, h2 = min(w2, image_width), min(h2, image_height)
        word_split_length = (w2 - w1) / len(annotation['label'])
        if w1 != w2 and h1 != h2 and h2 - h1 > 0 and w2 - w1 > 0:
            for i in range(len(annotation['label'])):
                wx = w1 + int(word_split_length * (i))
                if i == len(annotation['label']) - 1:
                    wy = wx + int(word_split_length)
                else:
                    wy = w2
                if wy - wx > 0:
                    if fill_method == 'solid':
                        region_heatmap[h1: h2, wx: wy] = get_solid_image(
                            image=np.zeros((h2 - h1, wy - wx))
                        )
                    else:
                        region_heatmap[h1: h2, wx: wy] = get_gaussian_image(
                            image=np.zeros((h2 - h1, wy - wx))
                        )
            if fill_method == 'solid':
                affinity_heatmap[h1: h2, w1: w2] = get_solid_image(
                    image=np.zeros((h2 - h1, w2 - w1))
                )
            else:
                affinity_heatmap[h1: h2, w1: w2] = get_gaussian_image(
                    image=np.zeros((h2 - h1, w2 - w1))
                )
    return np.reshape(
        np.concatenate((region_heatmap, affinity_heatmap), axis=0), 
        (2, image_height, image_width))