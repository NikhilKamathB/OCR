import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm


def get_annotated_file(image_path: str) -> str:
    '''
        Get annotated file path from image path.
        Input params:
            image_path: path to image file.
        Returns: annotated file path.
    '''
    assert image_path is not None, 'Image path is None. Provide a valid image path.'
    return image_path.replace('.jpg', '_ocr.json')

def normalize_image(image: np.ndarray) -> np.ndarray:
    '''
        Normalize image.
        Input params:
            image: np.ndarray of shape (image_height, image_width, channels[optional]).
        Returns: normalized image.
    '''
    return image / 255.0

def get_gaussian_image(image: np.ndarray, mean: float = 0.0, stddev: float = 0.5) -> np.ndarray:
    '''
        Get gaussian image.
        Input params:
            image: np.ndarray of shape (image_height, image_width, channels[optional]).
        Returns: gaussian image.
    '''
    x = np.linspace(-1, 1, image.shape[1])
    y = np.linspace(-1, 1, image.shape[0])
    x, y = np.meshgrid(x, y)
    image = norm.pdf(np.sqrt(x**2 + y**2), mean, stddev)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
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


def generate_region_affinity_heatmap(image_path: str) -> np.ndarray:
    '''
        Generate region heatmap for image.
        Input params:
            image_path: path to image file
        Returns: np.ndarray of shape (2, image_height, image_width)
    '''
    image_annotations_path = get_annotated_file(image_path)
    image = Image.open(image_path)
    image_width, image_height = image.size
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
        h1, h2 = min(annotation['y1'], annotation['y2']), \
            max(annotation['y3'], annotation['y4'])
        w1, w2 = min(annotation['x1'], annotation['x4']), \
            max(annotation['x2'], annotation['x3'])
        word_split_length = (w2 - w1) / len(annotation['label'])
        for i in range(len(annotation['label'])):
            wx = w1 + int(word_split_length * (i))
            if i == len(annotation['label']) - 1:
                wy = wx + int(word_split_length)
            else:
                wy = w2
            region_heatmap[h1: h2, wx: wy] = get_gaussian_image(
                image=np.zeros((h2 - h1, wy - wx))
            )
        affinity_heatmap[h1: h2, w1: w2] = get_gaussian_image(
            image=np.zeros((h2 - h1, w2 - w1))
        )
    return np.reshape(
        np.concatenate((region_heatmap, affinity_heatmap), axis=0), 
        (2, image_height, image_width))