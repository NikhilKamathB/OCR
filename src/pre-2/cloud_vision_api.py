import io
import os
import cv2
import random
import argparse
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from google.cloud import vision
from google.cloud.vision_v1 import types
from google.protobuf.json_format import MessageToDict
from config import config as conf


load_dotenv()

OUTPUT_FOLDER = 'api'

def str2bool(v):
    return v.lower() in ("yes", "Yes", "YES", "y", "true", "True", "TRUE", "t", "1")

def mk_dir(path=None):
    if not os.path.isdir(path):
        os.mkdir(path)
        os.mkdir(path+f'/{OUTPUT_FOLDER}')
    if not os.path.isdir(path+f'/{OUTPUT_FOLDER}'):
        os.mkdir(path+f'/{OUTPUT_FOLDER}')

def image_to_byte_array(image: Image, format: str):
    imgByteArr = io.BytesIO()
    if not format:
        image.save(imgByteArr, format=image.format)
    else:
        image.save(imgByteArr, format=format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

def convert_format(text_response):
    ls_word = [
        {
            'raw': text_response
        }
    ]
    texts = []
    if ('textAnnotations' in text_response):
        for text in text_response['textAnnotations']:
            boxes = {}
            boxes['label'] = text['description']
            boxes['x1'] = text['boundingPoly']['vertices'][0].get('x',0)
            boxes['y1'] = text['boundingPoly']['vertices'][0].get('y',0)
            boxes['x2'] = text['boundingPoly']['vertices'][1].get('x',0)
            boxes['y2'] = text['boundingPoly']['vertices'][1].get('y',0)
            boxes['x3'] = text['boundingPoly']['vertices'][2].get('x',0)
            boxes['y3'] = text['boundingPoly']['vertices'][2].get('y',0)
            boxes['x4'] = text['boundingPoly']['vertices'][3].get('x',0)
            boxes['y4'] = text['boundingPoly']['vertices'][3].get('y',0)
            boxes['w'] = boxes['x3'] - boxes['x1']
            boxes['h'] = boxes['y3'] - boxes['y1']
            texts.append(boxes)
    ls_word.append(texts)
    return ls_word

def google_vision_api(img):
    client = vision.ImageAnnotatorClient.from_service_account_file(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
    try:
        text_detection_response = client.document_text_detection(image=img)
        text_detection_response = MessageToDict(text_detection_response._pb)
    except Exception as e:
        text_detection_response = str(e)
        return text_detection_response
    responses = convert_format(text_detection_response)
    return responses

def gv_get_entities(image_bytes):
    try:
        image = types.Image(content=image_bytes)
    except Exception as e:
        return {'Error': str(e)}
    return google_vision_api(image)
    
def get_bboxes(image, format=None, verbose=True):
    ocr_response = []
    image_bytes = image_to_byte_array(image, format)
    ocr_response = gv_get_entities(image_bytes)
    if verbose:
        print(ocr_response[1])
    return ocr_response

def draw_lines(img, ocr_response, color=(255, 0, 0), thickness=2):
    img = np.array(img)
    for i in ocr_response:
        try:
            cv2.line(img, (i['x1'], i['y1']), (i['x2'], i['y2']), color, thickness)
            cv2.line(img, (i['x2'], i['y2']), (i['x3'], i['y3']), color, thickness)
            cv2.line(img, (i['x3'], i['y3']), (i['x4'], i['y4']), color, thickness)
            cv2.line(img, (i['x4'], i['y4']), (i['x1'], i['y1']), color, thickness)
            cv2.putText(img, i['label'], (i['x1'], i['y1']), fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=.5,color=(0, 181, 20, 255), thickness=1)
        except Exception as e:
            print(e)
    return img

def ocr(input_image=None, ouput_directory=None, draw_bbox=True):
    mk_dir(ouput_directory)
    image = Image.open(input_image)
    ocr_response = get_bboxes(image)
    image_cv2 = cv2.imread(input_image)[:, :, ::-1]
    #import pdb; pdb.set_trace()
    if draw_bbox:
        image = draw_lines(image_cv2, ocr_response[1])
        cv2.imwrite(os.path.join(ouput_directory, OUTPUT_FOLDER, input_image.split('/')[-1]), image[:,:,::-1])

def main(args=None):
    ocr(args.input_image, args.output_directory, args.draw_lines)

if __name__ == '__main__':
    # Fetching arguments.
    default_file = random.choice(os.listdir(conf.INTERIM_2021_06_14_ocr_kyc_pdfs_manual))
    parser = argparse.ArgumentParser(description='Cloud vision API to detect text.')
    parser.add_argument('-i', '--input_image', type=str, default=default_file, metavar="\b", help='Path to image')
    parser.add_argument('-o', '--output_directory', type=str, default=conf.DATA_DIR_OUTPUT, metavar="\b", help='Path to output location')
    parser.add_argument('-l', '--draw_lines', type=str2bool, default='y', metavar="\b", help='Draw lines on the image')
    args = parser.parse_args()
    main(args=args)
