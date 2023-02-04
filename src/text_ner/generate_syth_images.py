import os
from turtle import width
import cv2
import copy
import time
import json
import string
import random
import argparse
import numpy as np
from tqdm import tqdm
from config import config as conf


def str2bool(v):
    return v.lower() in ("yes", "Yes", "YES", "y", "true", "True", "TRUE", "t", "1")

def put_text(image, text, point, fontScale=[0.5, 0.75, 0.85], thickness=[1], font=[cv2.FONT_HERSHEY_SIMPLEX]):
    font_class = random.choice(font)
    thickness_class = random.choice(thickness)
    font_scale = random.choice(fontScale)
    label_size = cv2.getTextSize(text, font_class, fontScale=font_scale, thickness=thickness_class)
    image = cv2.putText(image, text, point, font_class, color=(0, 0, 0), fontScale=font_scale, thickness=thickness_class)
    text_list = [
        str(point[0]),
        str(point[1]-label_size[0][1]),
        str(point[0]+label_size[0][0]),
        str(point[1]-label_size[0][1]),
        str(point[0]+label_size[0][0]),
        str(point[1]),
        str(point[0]),
        str(point[1]),
        text
    ]
    return image, label_size, text_list

def render_text(image, point, value_font, text, space_pixel=10, line_pixel=25, fit=False):
    text_res = []
    initial_point = copy.deepcopy(point)
    for ind, text_line in enumerate(text):
        for text_item in text_line:
            image, label_size, text_list = put_text(image, text_item, initial_point, fontScale=value_font)
            text_res.append(text_list)
            initial_point[0] += label_size[0][0] + space_pixel
        initial_point = copy.deepcopy(point)
        initial_point[1] += (ind+1)*line_pixel
    return image, text_res, None if not fit else initial_point[1]

def get_image(resolution, resolution_factor, key_mapping, n=1, verbose=True, tolerance=3, output_dir=None, fit=False, img=None):
    for i in tqdm(range(n)):
        text_detections = []
        ie_dict = {k: '' for k in key_mapping.keys()}
        image_width = random.randint(resolution[0], resolution[1])
        image_height = int(image_width // resolution_factor)
        image = np.ones((image_height, image_width, 3)) * random.uniform(0.7, 1)
        next_y = None
        for key, value in key_mapping.items():
            key_point, value_point, key_font, value_font, value_type, max_value_length, start_with, max_words, max_lines = value
            if fit and next_y and value_point:
                point = (int(key_point[0]*image_width + random.choice([-1, 1])*random.randint(0, tolerance)), int(next_y + random.choice([-1, 1])*random.randint(0, tolerance)))
            else: 
                point = (int(key_point[0]*image_width + random.choice([-1, 1])*random.randint(0, tolerance)), int(key_point[1]*image_height + random.choice([-1, 1])*random.randint(0, tolerance)))
            text = key.upper()
            image, _, key_text_list = put_text(image, text, point, fontScale=key_font)
            if value_point:
                max_words = random.randint(1, max_words)
                if value_type == "number":
                    text = [[''.join(random.choices(string.digits, k=max_value_length)) for _ in range(max_words)] for _ in range(random.randint(1, max_lines))]
                elif value_type == "date":
                    text = [[''.join(random.choices(string.digits, k=2)) + '/' + ''.join(random.choices(string.digits, k=2)) + '/' + ''.join(random.choices(string.digits, k=4)) for _ in range(max_words)] for _ in range(random.randint(1, max_lines))]
                else:
                    text = [[''.join(random.choices(string.ascii_uppercase + string.digits, k=max_value_length)) for _ in range(max_words)] for _ in range(random.randint(1, max_lines))]
                text[0] = [start_with] + text[0]
                s = ''
                for ind, text_line in enumerate(text):
                    if ind == 0:
                        text_line_content = text_line[1:]
                    else:
                        text_line_content = text_line
                    s += ' '.join(text_line_content) + ' '
                ie_dict[key] = s.strip()
                if fit and next_y:
                    point = [int(value_point[0]*image_width + random.choice([-1, 1])*random.randint(0, tolerance)), int(next_y + random.choice([-1, 1])*random.randint(0, tolerance))]
                else: 
                    point = [int(value_point[0]*image_width + random.choice([-1, 1])*random.randint(0, tolerance)), int(value_point[1]*image_height + random.choice([-1, 1])*random.randint(0, tolerance))]
                image, value_text_list, next_y = render_text(image, point, value_font, text, fit=fit)
                text_detections += [i for i in value_text_list]
            text_detections += [key_text_list]
        if img and isinstance(img, dict):
            top_left = [int(img['top_left'][0]*image_width + random.choice([-1, 1])*random.randint(0, tolerance)), int(img['top_left'][1]*image_height + random.choice([-1, 1])*random.randint(0, tolerance))]
            bottom_right = [int(img['bottom_right'][0]*image_width + random.choice([-1, 1])*random.randint(0, tolerance)), int(img['bottom_right'][1]*image_height + random.choice([-1, 1])*random.randint(0, tolerance))]
            np_img = np.random.rand((bottom_right[1]-top_left[1]), (bottom_right[0]-top_left[0]), 3)*255
            np_img = np_img.astype(int)
            image[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0], :] = np_img
        if output_dir:
            file_name = str(int(time.time() * 1000000)) + f'_{i+1}'
            with open(os.path.join(output_dir, 'txt', file_name+'.txt'), 'w') as f:
                for detection in text_detections:
                    f.write(','.join(detection) + '\n')
            with open(os.path.join(output_dir, 'json',  file_name+'.json'), 'w') as f:
                json.dump(ie_dict, f)
            cv2.imwrite(os.path.join(output_dir, 'images', file_name+'.jpg'), image*255.0)
        if verbose:
            cv2.imshow("Image - synthetic", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main(args=None):
    if args.stnk:
        get_image(conf.STNK_SYNTH_RES, conf.STNK_X_Y_RATIO, conf.STNK_SYNTH, args.num_images, args.verbose, conf.TOLERANCE, args.output_dir)
    elif args.ktp:
        get_image(conf.KTP_SYNTH_RES, conf.KTP_X_Y_RATIO, conf.KTP_SYNTH, args.num_images, args.verbose, conf.TOLERANCE, args.output_dir, args.fit, conf.KTP_IMAGE)

if __name__ == '__main__':
    # Fetching arguments.
    parser = argparse.ArgumentParser(description='Remove duplicate files.')
    parser.add_argument('-n', '--num_images', type=int, default=0, metavar="\b", help='Number of images to generate')
    parser.add_argument('-o', '--output_dir', type=str, default=None, metavar="\b", help='Output Dir')
    parser.add_argument('-v', '--verbose', type=str2bool, default='y', metavar="\b", help='Verbose')
    parser.add_argument('-f', '--fit', type=str2bool, default='y', metavar="\b", help='Fit image? (KTP and SIM)')
    parser.add_argument('-stnk', '--stnk', type=str2bool, default='y', metavar="\b", help='Generate STNK documents?')
    parser.add_argument('-ktp', '--ktp', type=str2bool, default='n', metavar="\b", help='Generate KTP documents?')
    args = parser.parse_args()
    main(args=args)