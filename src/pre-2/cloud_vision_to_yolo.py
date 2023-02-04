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
from google.protobuf.json_format import MessageToDict, MessageToJson
from config import config as conf


load_dotenv()

def str2bool(v):
    return v.lower() in ("yes", "Yes", "YES", "y", "true", "True", "TRUE", "t", "1")

def mk_dir(path=None):
    if not os.path.isdir(path):
        os.mkdir(path)

def image_to_byte_array(image: Image, format: str):
    imgByteArr = io.BytesIO()
    if not format:
        image.save(imgByteArr, format=image.format)
    else:
        image.save(imgByteArr, format=format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

def convert_format(text_response):
    bboxs = []
    # import pdb; pdb.set_trace()
    imgw, imgh = text_response['fullTextAnnotation']['pages'][0]['width'], text_response['fullTextAnnotation']['pages'][0]['height']

    if ('textAnnotations' in text_response):
        for text in text_response['textAnnotations']:
            boxes = {}
           
            x1 = text['boundingPoly']['vertices'][0].get('x',0)
            y1 = text['boundingPoly']['vertices'][0].get('y',0)
            x2 = text['boundingPoly']['vertices'][1].get('x',0)
            y2 = text['boundingPoly']['vertices'][1].get('y',0)
            x3 = text['boundingPoly']['vertices'][2].get('x',0)
            y3 = text['boundingPoly']['vertices'][2].get('y',0)
            x4 = text['boundingPoly']['vertices'][3].get('x',0)
            y4 = text['boundingPoly']['vertices'][3].get('y',0)
            

            xmin = min(x1, x4)
            xmax = max(x2, x3)
            cx = (xmax + xmin) // 2

            ymin = min(y1, y2)
            ymax = max(y3, y4)
            cy = (ymax + ymin) // 2

            center = ( cx/imgw , cy/imgh )
            width = (x2-x1)/imgw
            height = (y3-y2)/imgh

            bboxs.append((*center, width, height))

            
    return bboxs

def google_vision_api(img):
    client = vision.ImageAnnotatorClient.from_service_account_file(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
    try:
        text_detection_response = client.document_text_detection(image=img)
        json_detection_response = MessageToJson(text_detection_response._pb)
        text_detection_response = MessageToDict(text_detection_response._pb)
    except Exception as e:
        text_detection_response = str(e)
        return text_detection_response
    responses = convert_format(text_detection_response)
    return responses, json_detection_response

def gv_get_entities(image_bytes):
    try:
        image = types.Image(content=image_bytes)
    except Exception as e:
        return {'Error': str(e)}
    return google_vision_api(image)
    
def get_bboxes(image, format=None, verbose=True):
    ocr_response = []
    image_bytes = image_to_byte_array(image, format)
    ocr_response, ocr_json = gv_get_entities(image_bytes)
    if verbose:
        print(ocr_response[1])
    return ocr_response, ocr_json

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

def ocr(input_image=None, output_directory=None, draw_bbox=True):
    mk_dir(output_directory)
    image = Image.open(input_image)
    ocr_response, ocr_json = get_bboxes(image)
    image_cv2 = cv2.imread(input_image)[:, :, ::-1]
    if draw_bbox:
        image = draw_lines(image_cv2, ocr_response[1])
        cv2.imwrite(os.path.join(output_directory, input_image.split('/')[-1]), image[:,:,::-1])
    
    json_out_filename = input_image.split('/')[-1].split('.')[-2] + '.json'
    with open(os.path.join(output_directory, 'json_real', json_out_filename), "w") as wf:
        wf.write(ocr_json)

    txt_out_filename = input_image.split('/')[-1].split('.')[-2] + '.txt'
    with open(os.path.join(output_directory, 'txt_real', txt_out_filename), "w") as wf:
        for bbox in ocr_response:
            wf.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


def main(args=None):
    ocr(args.input_image, args.output_directory, args.draw_lines)

if __name__ == '__main__':
    # Fetching arguments.
    # default_file = random.choice(os.listdir(conf.INTERIM_2021_06_14_ocr_kyc_pdfs_manual))
    parser = argparse.ArgumentParser(description='Cloud vision API to detect text.')
    parser.add_argument('-i', '--input_image', type=str, metavar="\b", help='Path to image')
    parser.add_argument('-o', '--output_directory', type=str, default=conf.DATA_DIR_OUTPUT, metavar="\b", help='Path to output location')
    parser.add_argument('-l', '--draw_lines', type=str2bool, default='y', metavar="\b", help='Draw lines on the image')
    args = parser.parse_args()
    main(args=args)
