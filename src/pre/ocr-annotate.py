import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from google.cloud import vision
from google.cloud.vision_v1 import types
from google.protobuf.json_format import MessageToDict
from utils import *
from config import config as conf


load_dotenv()

def convert_format(text_response, item):
    texts = []
    image_width, image_height = text_response['fullTextAnnotation']['pages'][0]['width'], text_response['fullTextAnnotation']['pages'][0]['height']
    if ('textAnnotations' in text_response):
        for text in text_response['textAnnotations']:
            bboxes = {}
            bboxes['label'] = text['description']
            bboxes['x1'] = text['boundingPoly']['vertices'][0].get('x',0)
            bboxes['y1'] = text['boundingPoly']['vertices'][0].get('y',0)
            bboxes['x2'] = text['boundingPoly']['vertices'][1].get('x',0)
            bboxes['y2'] = text['boundingPoly']['vertices'][1].get('y',0)
            bboxes['x3'] = text['boundingPoly']['vertices'][2].get('x',0)
            bboxes['y3'] = text['boundingPoly']['vertices'][2].get('y',0)
            bboxes['x4'] = text['boundingPoly']['vertices'][3].get('x',0)
            bboxes['y4'] = text['boundingPoly']['vertices'][3].get('y',0)
            xmin = min(bboxes['x1'], bboxes['x4'])
            xmax = max(bboxes['x2'], bboxes['x3'])
            ymin = min(bboxes['y1'], bboxes['y2'])
            ymax = max(bboxes['y3'], bboxes['y4'])
            cx = (xmax + xmin) // 2
            cy = (ymax + ymin) // 2
            bboxes['cx'], bboxes['cy'] = cx, cy
            bboxes['bbox_w'] = xmax - xmin
            bboxes['bbox_h'] = ymax - ymin
            bboxes['cx_yolo'], bboxes['cy_yolo'] = cx / image_width, cy / image_height
            bboxes['bbox_w_yolo'] = (xmax - xmin) / image_width
            bboxes['bbox_h_yolo'] = (ymax - ymin) / image_height
            texts.append(bboxes)
    annotations = {
        'image': item,
        'image_width': image_width,
        'image_height': image_height,
        'annotations': texts,
        'raw': text_response
    }
    return annotations

def google_vision_api(image, item):
    client = vision.ImageAnnotatorClient.from_service_account_file(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
    try:
        text_detection_response = client.document_text_detection(image=image)
        text_detection_response = MessageToDict(text_detection_response._pb)
    except Exception as e:
        text_detection_response = str(e)
        return text_detection_response
    response = convert_format(text_detection_response, item)
    return response

def gv_get_entities(image_bytes, item):
    try:
        image = types.Image(content=image_bytes)
    except Exception as e:
        return {'Error': str(e)}
    return google_vision_api(image, item)
    
def get_bboxes(image, item, format=None):
    image_bytes = image_to_byte_array(image, format)
    ocr_response = gv_get_entities(image_bytes, item)
    return ocr_response

def ocr_engine(input_directory=None, ouput_directory=None):
    print(f'Input directory - {input_directory}\nOutput directory - {ouput_directory}')
    mk_dir(ouput_directory)
    for item in tqdm([_ for _ in os.listdir(input_directory) if '.jpg' in _.lower() or '.png' in _.lower() or '.jpeg' in _.lower()]):
        file_name = item.split('.')[0]
        image = Image.open(os.path.join(input_directory, item))
        ocr_response = get_bboxes(image, item)
        with open(os.path.join(ouput_directory, file_name + '.json'), "w") as f: 
            json.dump(ocr_response, f)
        with open(os.path.join(ouput_directory, file_name + '.txt'), "w") as f:
            for detection in ocr_response['annotations']:
                f.write(f"0 {detection['cx_yolo']} {detection['cy_yolo']} {detection['bbox_w_yolo']} {detection['bbox_h_yolo']}\n")
    with open(os.path.join(ouput_directory, 'classes.txt'), "w") as f:
        for label in conf.LABEL:
            f.write(f"{label}\n")

def main(args=None):
    ocr_engine(args.input_directory, args.output_directory)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Google cloud vision API - OCR.')
    parser.add_argument('-i', '--input_directory', type=str, default=conf.DATA_DIR_RAW_TRAIN, metavar="\b", help='Input image directory')
    parser.add_argument('-o', '--output_directory', type=str, default=conf.DATA_DIR_ANNOTATION_TRAIN, metavar="\b", help='Output directory')
    args = parser.parse_args()
    main(args=args)