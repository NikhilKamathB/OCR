import cv2
import json
import copy
import argparse
import pandas as pd
from cloud_vision_api import *
from config import config as conf


def draw_bboxes(image, annotations, color=(255, 0, 0), thickness=2):
    image = np.array(image)
    for i in annotations:
        try:
            cv2.rectangle(image, (i['top_left'][0], i['top_left'][1]), (i['bottom_right'][0], i['bottom_right'][1]), color, thickness)
        except Exception as e:
            print(e)
    return image

def visualize(csv_path, sample_count):
    df = pd.read_csv(csv_path)
    df_sample = df.sample(n=sample_count)
    for _, row in df_sample.iterrows():
        image = cv2.imread(row['image_path'])[:, :, ::-1]
        image_copy = copy.deepcopy(image)
        with open(row['json_path']) as f:
            annotations = json.load(f)
        image_copy = draw_bboxes(image_copy, annotations['annotations'])
        cv2.imshow("Image - True", image)
        cv2.imshow("Image - Annotated", image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main(args=None):
    visualize(args.annotation_csv_file, args.sample_count)

if __name__ == '__main__':
    # Fetching arguments.
    parser = argparse.ArgumentParser(description='Annotate images')
    parser.add_argument('-f', '--annotation_csv_file', type=str, default=conf.INTERIM_2021_06_14_ocr_kyc_pdfs_annotations_csv, metavar="\b", help='Path to directory from which images will be annotated')
    parser.add_argument('-c', '--sample_count', type=int, default=1, metavar="\b", help='Number of items you would like to check')
    args = parser.parse_args()
    main(args=args)