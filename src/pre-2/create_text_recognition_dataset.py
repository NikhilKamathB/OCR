import os
import re
import cv2
import json
import time
import shutil
import random
import string
import argparse
import pandas as pd
from tqdm import tqdm
from config import config as conf


random_string_length = 5

def mk_dir(path=None):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)

def crop(image, x1, y1, x2, y2):
    return image[y1:y2, x1:x2]

def generate_data(save_path, folder_name, image_path, json_path, threshold_pixel=5):
    file_name_list, file_path_list, labels = [], [], []
    image = cv2.imread(image_path)
    with open(json_path) as f:
        annotations = json.load(f)
    detections = [i for i in annotations['annotations'] if len(i['text'].replace('\n', ' ').split()) == 1]
    for detection in detections:
        x1, y1, x2, y2 = max(0, detection['top_left'][0]), max(0, detection['top_left'][1]), min(image.shape[1], detection['bottom_right'][0]), min(image.shape[0], detection['bottom_right'][1])
        if x2-x1 > threshold_pixel and y2-y1 > threshold_pixel:
            cropped_image = crop(image, x1, y1, x2, y2)
            random_string = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(random_string_length))
            text_abstract = re.sub('[^a-zA-Z0-9]+', '', detection['text'].lower())
            if not text_abstract:
                text_abstract = 'special_character'
            file_name = f"{folder_name}___{text_abstract}___{int(time.time()*1000)}_{random_string}.jpg"
            file_path = os.path.join(save_path, file_name)
            try:
                saved = cv2.imwrite(r'{file_path}'.format(file_path=file_path), cropped_image)
                if saved:
                    labels.append(detection['text'].lower())
                    file_name_list.append(file_name)
                    file_path_list.append(file_path)
            except Exception as e:
                print(e)
                continue
    return file_name_list, file_path_list, labels

def extract(csv_path, output_path, threshold_pixel):
    df_dict = {
        'folder_name': [],
        'image_path': [],
        'image_file_name': [],
        'label': [], 
        'parent_image_path': [],
        'parent_json_path': []
    }
    annots = pd.read_csv(csv_path)
    mk_dir(output_path)
    for _, row in tqdm(annots.iterrows()):
        folder_name = row['file_name']
        folder_path = os.path.join(output_path, folder_name)
        mk_dir(folder_path)
        file_name_list, file_path_list, labels = generate_data(folder_path, folder_name, row['image_path'], row['json_path'], threshold_pixel)
        instances_created = len(labels)
        if instances_created > 0:
            df_dict['folder_name'] += [folder_name] * instances_created
            df_dict['image_path'] += file_path_list
            df_dict['image_file_name'] += file_name_list
            df_dict['label'] += labels
            df_dict['parent_image_path'] += [row['image_path']] * instances_created
            df_dict['parent_json_path'] += [row['json_path']] * instances_created
    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(output_path, 'text_recognition_annots.csv'))

def main(args=None):
    extract(args.annotation_csv_file, args.output_directory, args.threshold_pixel)

if __name__ == '__main__':
    # Fetching arguments.
    parser = argparse.ArgumentParser(description='Annotate images')
    parser.add_argument('-f', '--annotation_csv_file', type=str, default=conf.INTERIM_2021_06_14_ocr_kyc_pdfs_annotations_csv, metavar="\b", help='Path to directory from which images will be annotated')
    parser.add_argument('-o', '--output_directory', type=str, default=conf.INTERIM_2021_08_10_text_recognition, metavar="\b", help='Output directory to store the text recognition dataset')
    parser.add_argument('-t', '--threshold_pixel', type=int, default=5, metavar="\b", help='Threshold value for considering data based on either the difference between widths or heights')
    args = parser.parse_args()
    main(args=args)