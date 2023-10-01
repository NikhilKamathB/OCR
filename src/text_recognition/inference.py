import os
import json
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image
from transformers import TrOCRProcessor
try: 
    from .net import OCRModel
except Exception as e:
    print("Following alternate import for inference.py...")
    from net import OCRModel


class Inference:

    '''
        This class is used to perform inference on the model.
    '''

    def __init__(self,
                 saved_model: str,
                 input_image_directory: str,
                 number_of_images_to_infer: int = 1,
                 shuffle: bool = False,
                 wrtie_output: bool = False,
                 output_directory: str = None,
                 model_name: str = "TrOCRModel",
                 device: str = "cpu",
                 processor_pretrained_path: str = "microsoft/trocr-base-handwritten",
                 verbose: bool = True) -> None:
        '''
            Initialize the inference class.
            Input params: 
                saved_model: Path to saved model.
                input_image_directory: Path to input image directory.
                number_of_images_to_infer: Total number of images to process.
                shuffle: Shuffle images in the aforementioned directory.
                wrtie_output: Write output to disk - yes or no.
                output_directory: Path to output directory that holds results.
                model_name: Name of the model.
                device: Device to use | 'cpu', 'cuda', 'mps'.
                processor_pretrained_path: Path to pretrained processor.
                verbose: Print verbose statements.
        '''
        self.write_output = wrtie_output
        self.output_directory = output_directory
        self.verbose = verbose
        print("Fetching model...")
        self.device = device
        self.module = OCRModel(
            device=self.device,
            model_name=model_name,
            saved_model=saved_model,
            freeze_model=True,
            verbose=False
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
        self.processor = TrOCRProcessor.from_pretrained(processor_pretrained_path)

    def process_image(self, image_path: str) -> dict:
        '''
            Process the image.
            Input params:
                image_path: Path to image.
            Output params:
                detections: Bounding box for the image.
        '''
        print(f"\nProcessing image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = self.processor(image, return_tensors="pt").pixel_values
        output = self.model.generate(image.to(self.device))
        output = output.cpu().detach().numpy()
        output = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        if self.verbose:
            print(f"Text: {output}")
            plt.imshow(Image.open(image_path).convert("RGB"))
            plt.show()
        return {
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
            "text": output
        }
    
    def infer(self) -> None:
        '''
            Infer on the images.
        '''
        print("Inferencing...")
        for image_path in self.images:
            detections = self.process_image(image_path=image_path)
            # Write the output to a json file.
            if self.write_output:
                os.makedirs(self.output_directory, exist_ok=True)
                with open(os.path.join(self.output_directory, os.path.basename(image_path).split(".")[0] + ".json"), "w") as f:
                    json.dump(detections, f, indent=4)