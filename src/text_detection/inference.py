import os
import cv2
import json
import glob
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from net import OCRModel
from utils import normalize_image


class Inference:

    def __init__(self,
                 saved_model: str,
                 input_image_directory: str,
                 number_of_images_to_infer: int = 1,
                 resize: tuple = (768, 768),
                 shuffle: bool = False,
                 wrtie_output: bool = False,
                 output_directory: str = None,
                 model_name: str = "craft",
                 device: str = "cpu",
                 cv2_threshold_low: int = 100,
                 cv2_threshold_high: int = 255,
                 cv2_dilate_kernel_size: tuple = (3, 3),
                 cv2_dilate_iteration: int = 1,
                 detection_buffer: int = 2,
                 verbose: bool = True) -> None:
        self.write_output = wrtie_output
        self.output_directory = output_directory
        self.cv2_threshold_low = cv2_threshold_low
        self.cv2_threshold_high = cv2_threshold_high
        self.cv2_dilate_kernel_size = cv2_dilate_kernel_size
        self.cv2_dilate_iteration = cv2_dilate_iteration
        self.detection_buffer = detection_buffer
        self.verbose = verbose
        print("Fetching model...")
        self.module = OCRModel(
            device=device,
            model_name=model_name,
            saved_model=saved_model,
            freeze_model=True,
            raw_load=True
        )
        self.model = self.module.model
        self.model.eval()
        print("Fetching images...")
        images = glob.glob(input_image_directory + "/*.jpg") + \
                    glob.glob(input_image_directory + "/*.jpeg") + \
                    glob.glob(input_image_directory + "/*.png")
        if shuffle:
            random.shuffle(images)
        self.images = images[0: min(number_of_images_to_infer, len(images))]
        self.resize_tuple = resize

    def resize(self, image: object, size: tuple = (768, 768)) -> np.ndarray:
        return np.asarray(image.resize((size[1], size[0])))

    def normalize(self, image: np.ndarray) -> np.ndarray:
        return normalize_image(image=image)

    def get_bbox(self, image_path: str, region_map: np.ndarray) -> dict:
        region_map *= 255.0
        region_map = region_map.astype(np.uint8)    
        _, region_map = cv2.threshold(region_map, self.cv2_threshold_low, self.cv2_threshold_high, cv2.THRESH_BINARY)
        region_map_dilated = cv2.dilate(region_map, np.ones(self.cv2_dilate_kernel_size, np.uint8), iterations=self.cv2_dilate_iteration)
        region_map_contours, region_map_hierarchy = cv2.findContours(region_map_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = {}
        for idx, contour in enumerate(region_map_contours):
            if contour.shape[0] <= 1:
                continue
            x_min, y_min = np.inf, np.inf
            x_max, y_max = -np.inf, -np.inf
            for point in contour:
                x_min = min(x_min, point[0][0])
                y_min = min(y_min, point[0][1])
                x_max = max(x_max, point[0][0])
                y_max = max(y_max, point[0][1])
            detections[f"bbox_{idx+1}"] = { "actual_bbox": (
                    (x_min - self.detection_buffer, y_min - self.detection_buffer),
                    (x_max + self.detection_buffer, y_max + self.detection_buffer)
                ),
                "normalized_bbox": (
                    ((x_min - self.detection_buffer)/region_map.shape[1], (y_min - self.detection_buffer)/region_map.shape[0]),
                    ((x_max + self.detection_buffer)/region_map.shape[1], (y_max + self.detection_buffer)/region_map.shape[0])
                )
            }
        if self.verbose:
            image = Image.open(image_path).convert("RGB")
            image = self.resize(image=image, size=(region_map.shape[0], region_map.shape[1]))
            for _, v in detections.items():
                x1, y1 = int(v["normalized_bbox"][0][0] * region_map.shape[1]), int(v["normalized_bbox"][0][1] * region_map.shape[0])
                x2, y2 = int(v["normalized_bbox"][1][0] * region_map.shape[1]), int(v["normalized_bbox"][1][1] * region_map.shape[0])
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plt.imshow(image)
            plt.show()
        return detections

    def process_image(self, image_path: str) -> dict:
        print(f"Processing image: {image_path}")
        # Get the image in model readable format.
        image = Image.open(image_path).convert("RGB")
        image = self.resize(image=image, size=self.resize_tuple)
        image = self.normalize(image=image)
        image = torch.from_numpy(image) # image shape - h, w, c
        image = image.unsqueeze(0) # image shape - 1, h, w, c
        image = image.permute(0, 3, 1, 2) # image shape - 1, c, h, w
        image = torch.autograd.Variable(image)
        # Feed this image into the model.
        output, _ = self.model(image)
        output = output.cpu().detach().numpy()
        region_map, _ = output[0, :, :, 0], output[0, :, :, 1]
        # Get bounding boxes from the output for this image.
        detections = self.get_bbox(image_path=image_path, region_map=region_map)
        return detections
    
    def infer(self) -> None:
        print("Inferencing...")
        for image_path in self.images:
            detections = self.process_image(image_path=image_path)
            # Write the output to a json file.
            if self.write_output:
                os.makedirs(self.output_directory, exist_ok=True)
                with open(os.path.join(self.output_directory, os.path.basename(image_path).split(".")[0] + ".json"), "w") as f:
                    json.dump(detections, f, indent=4)
