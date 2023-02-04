import os
import io
import cv2
import random
import argparse
from PIL import Image
import numpy as np
from google.cloud import vision
from google.cloud.vision_v1 import types
from google.protobuf.json_format import MessageToDict
from scipy.ndimage import interpolation as inter
from ner import *



def str2bool(v):
    return v.lower() in ("yes", "Yes", "YES", "y", "true", "True", "TRUE", "t", "1")

def view_bbox_true(image, detections, color=(0, 0, 255), thickness=1):
    image = np.array(image)
    for i in detections:
        try:
            cv2.rectangle(image, (i['x1'], i['y1']), (i['x3'], i['y3']), color, thickness)
        except Exception as e:
            print(e)
    return image

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
    
def get_bboxes(image, format=None, verbose=False):
    ocr_response = []
    image_bytes = image_to_byte_array(image, format)
    ocr_response = gv_get_entities(image_bytes)
    return ocr_response

def process_doc(image=None, verbose=True):
    morph_kernel = np.ones((5, 5))
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img = cv2.imread(image)
    # img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)

    # img_dup = img.copy()
    # img_canny = cv2.Canny(img_dup, 100, 150)
    # img_dilate = cv2.dilate(img_canny, morph_kernel, iterations=2)
    # img_erode = cv2.erode(img_dilate, morph_kernel)
    # contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # x_min, y_min = np.inf, np.inf
    # x_max, y_max = 0, 0
    # for c in contours:
    #     for coord in c:
    #         if coord[0][0] < x_min:
    #             x_min = coord[0][0]
    #         if coord[0][0] > x_max:
    #             x_max = coord[0][0]
    #         if coord[0][1] < y_min:
    #             y_min = coord[0][1]
    #         if coord[0][1] > y_max:
    #             y_max = coord[0][1]
    # # print((x_min, y_min), '   ', (x_max, y_max))
    # p1 = np.float32([[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]])
    # p2 = np.float32([[0, 0], [img_dup.shape[1], 0], [0, img_dup.shape[0]], [img_dup.shape[1], img_dup.shape[0]]])
    # mtx = cv2.getPerspectiveTransform(p1, p2)
    # res = cv2.warpPerspective(img_dup, mtx, (img_dup.shape[1], img_dup.shape[0]))
    
    # img = cv2.resize(img, dsize=(max(img.shape[0], 720), max(img.shape[1], 512)), interpolation=cv2.INTER_LANCZOS4)
    # image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=sharpen_kernel)
    if verbose:
        cv2.imshow("Image - Org", img)
        cv2.imshow("Image - Processed", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # image_pil = Image.open(image)
    # image_pil = Image.fromarray(image_sharp)
    image_pil = Image.fromarray(img)
    ocr_response = get_bboxes(image_pil, format='JPEG')
    detections = ocr_response[1]
    if verbose:
        image_actual = view_bbox_true(img, detections)
        cv2.imshow("Image - OCR engine output", image_actual)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main(args=None):
    images = [os.path.join(dir, img) for dir in args.input_directory for img in os.listdir(dir)]
    random.shuffle(images)
    for ind, image in enumerate(images):
        if 'image.png' in image:
            print(f"{ind+1}. Processing {image} ...")
            process_doc(image, args.view_output)
            if ind+1 == args.num_images:
                break


if __name__ == '__main__':
    # Fetching arguments.
    parser = argparse.ArgumentParser(description='Remove duplicate files.')
    parser.add_argument('-i', '--input_directory', nargs='+', metavar="\b", help='Image dir')
    parser.add_argument('-v', '--view_output', type=str2bool, default='y', metavar="\b", help='View output?')
    parser.add_argument('-n', '--num_images', type=int, default=0, metavar="\b", help='Number of images to process')
    args = parser.parse_args()
    main(args=args)