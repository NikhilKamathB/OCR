import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../text_detection_craft')

import os
import cv2
import math
import random
import argparse
import numpy as np
from PIL import Image
from config import config as conf
from cloud_vision_api import *
from text_detection_craft import run as craft_run
from text_detection_craft import utils as craft_utils


def str2bool(v):
    return v.lower() in ("yes", "Yes", "YES", "y", "true", "True", "TRUE", "t", "1")

def calculate_iou(bbox_true=None, bbox_pred=None):
    x1 = max(bbox_true[0], bbox_pred[0])
    y1 = max(bbox_true[1], bbox_pred[1])
    x2 = min(bbox_true[2], bbox_pred[2])
    y2 = min(bbox_true[3], bbox_pred[3])
    intersection_area = max(0, x2-x1+1) * max(0, y2-y1+1)
    bbox_true_area = (bbox_true[2]-bbox_true[0]+1) * (bbox_true[3]-bbox_true[1]+1)
    bbox_pred_area = (bbox_pred[2]-bbox_pred[0]+1) * (bbox_pred[3]-bbox_pred[1]+1)
    iou = intersection_area / float(bbox_true_area + bbox_pred_area - intersection_area)
    return iou

def get_ground_truth(image, format=None, verbose=False):
    ocr_response = get_bboxes(image, format, verbose=False)
    return ocr_response[1]

def view_bbox_true(image, ocr_response, color=(255, 0, 0), thickness=2):
    image = np.array(image)
    for i in ocr_response:
        try:
            cv2.line(image, (i['x1'], i['y1']), (i['x2'], i['y2']), color, thickness)
            cv2.line(image, (i['x2'], i['y2']), (i['x3'], i['y3']), color, thickness)
            cv2.line(image, (i['x3'], i['y3']), (i['x4'], i['y4']), color, thickness)
            cv2.line(image, (i['x4'], i['y4']), (i['x1'], i['y1']), color, thickness)
        except Exception as e:
            print(e)
    return image

def view_bbox_pred(image, boxes):
    image = np.array(image)
    for _, box in enumerate(boxes):
        try:
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            cv2.polylines(image, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
        except Exception as e:
            print(e)
    return image

def get_recognitions_craft(image, bboxes, ocr_response_processed, vertical_padding_percentage=0.1, horizontal_padding_percentage=0.05):
    recognitions = []
    print('Recognizing words...')
    for ind, box in enumerate(bboxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        bbox = poly.reshape((-1, 2))
        top = max(0, int(bbox[0][0]-horizontal_padding_percentage*bbox[0][0]))
        right = max(0, int(bbox[0][1]-vertical_padding_percentage*bbox[0][1]))
        left = min(image.size[0], int(bbox[2][0]+horizontal_padding_percentage*bbox[2][0]))
        bottom = min(image.size[1], int(bbox[2][1]+vertical_padding_percentage*bbox[2][1]))
        image_ref = image.crop((top, right, left, bottom))
        ocr_response = get_ground_truth(image_ref, image.format)
        word = ocr_response[-1]['label']
        same_words = [i for i in ocr_response_processed if word == i['label']]
        same_words_count = len(same_words)
        if same_words_count == 1:
            recognitions.append(
                {   
                    'true_value': ([same_words[0]['x1'], same_words[0]['y1'], same_words[0]['x3'], same_words[0]['y3']], same_words[0]['label']),
                    'predicted_value': ([bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]], ocr_response[-1]['label'])
                }
            )
        elif same_words_count > 1:
            x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]
            min_dist = np.inf
            true_parent = None
            for i in same_words:
                x_1 = math.pow(x1-i['x1'], 2)
                y_1 = math.pow(y1-i['y1'], 2)
                x_2 = math.pow(x2-i['x3'], 2)
                y_2 = math.pow(y2-i['y3'], 2)
                dist = math.pow(x_1+y_1, 0.5) + math.pow(x_2+y_2, 0.5)
                if dist < min_dist:
                    min_dist = dist
                    true_parent = i
            recognitions.append(
                {   
                    'true_value': ([true_parent['x1'], true_parent['y1'], true_parent['x3'], true_parent['y3']], true_parent['label']),
                    'predicted_value': ([bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]], ocr_response[-1]['label'])
                }
            )
        else:
            continue
        print(f'Word {ind+1} recorded -> {word}')
    return recognitions

def compute_iou_craft(recognitions, input_image):
    print('Computing IOU...')
    iou = 0
    for recognition in recognitions:
        iou += calculate_iou(recognition['true_value'][0], recognition['predicted_value'][0])
    print(f'IOU for detected texts in the image {input_image} = {iou}')

def get_craft_iou(trained_model_path, device, canvas_size, text_threshold, link_threshold, low_text, mag_ratio, input_image=None, view_output=True, vertical_padding_percentage=0.1, horizontal_padding_percentage=0.05):
    model = craft_run.load(trained_model_path, device)
    model.eval()
    image = Image.open(input_image)
    ocr_response = get_ground_truth(image)
    ocr_response_processed = [i for i in ocr_response if len(i['label'].replace('\n', ' ').split()) == 1]
    image_cv2 = craft_utils.loadImage(input_image)
    bboxes, _ = craft_run.detect_text(model, image_cv2, canvas_size, text_threshold, link_threshold, low_text, mag_ratio, device)
    recognitions = get_recognitions_craft(image, bboxes, ocr_response_processed, vertical_padding_percentage, horizontal_padding_percentage)
    compute_iou_craft(recognitions, input_image)
    if view_output:
        image_true = view_bbox_true(image_cv2, ocr_response_processed)
        image_pred = view_bbox_pred(image_cv2, bboxes)
        image = view_bbox_pred(image_true, bboxes)
        cv2.imshow("Image - True", image_true)
        cv2.imshow("Image - Predicted", image_pred)
        cv2.imshow("Image - Combined", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
def main(args=None):
    device = 'cpu' if not args.gpu else conf.DEVICE
    if args.model_name.strip().lower() == 'craft':
        get_craft_iou(args.trained_model, device, args.canvas_size, args.text_threshold, args.link_threshold, args.low_text, args.mag_ratio, args.input_image, args.view_output, args.vertical_padding_percentage, args.horizontal_padding_percentage)

if __name__ == '__main__':
    # Fetching arguments.
    default_file = random.choice(os.listdir(f'../{conf.INTERIM_2021_06_14_ocr_kyc_pdfs}'))
    parser = argparse.ArgumentParser(description='Remove duplicate files.')
    parser.add_argument('-m', '--trained_model', type=str, default=f'../{conf.CRAFT}', metavar="\b", help='Path to trained model')
    parser.add_argument('-n', '--model_name', type=str, default='craft', metavar="\b", help='Name of the model used to detect text')
    parser.add_argument('-i', '--input_image', type=str, default=default_file, metavar="\b", help='Path to image')
    parser.add_argument('-vp', '--vertical_padding_percentage', default=0.1, type=float, metavar="\b", help='Vertical padding percentage')
    parser.add_argument('-hp', '--horizontal_padding_percentage', default=0.05, type=float, metavar="\b", help='Horizontal padding percentage')
    parser.add_argument('-text', '--text_threshold', default=0.7, type=float, metavar="\b", help='Text confidence threshold')
    parser.add_argument('-low', '--low_text', default=0.4, type=float, metavar="\b", help='Text low-bound score')
    parser.add_argument('-lnk', '--link_threshold', default=0.4, type=float, metavar="\b", help='Link confidence threshold')
    parser.add_argument('-c', '--canvas_size', default=1280, type=int, metavar="\b", help='Image size for inference')
    parser.add_argument('-mr', '--mag_ratio', default=1.5, type=float, metavar="\b", help='Image magnification ratio')
    parser.add_argument('-g', '--gpu', type=str2bool, default='n', metavar="\b", help='Use GPU?')
    parser.add_argument('-v', '--view_output', type=str2bool, default='y', metavar="\b", help='View output?')
    args = parser.parse_args()
    main(args=args)