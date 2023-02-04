from operator import index
import os
import re
import cv2
import json
import copy
import random
import argparse
import numpy as np
import pandas as pd
from tomlkit import key
from ner_stnk import *
from config import config as conf


def str2bool(v):
    return v.lower() in ("yes", "Yes", "YES", "y", "true", "True", "TRUE", "t", "1")

def view_bbox_true(image, detections, color=(0, 0, 255), association_color=(255, 0, 0), grouped_association_color=(0, 255, 0), thickness=1, association_thickness=1, grouped_association_thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_color=(255, 0, 0), associated_font_color=(0, 255, 0), font_thickness=1, printing_association=False):
    image = np.array(image)
    for i in detections:
        try:
            if not printing_association:
                cv2.rectangle(image, (i['top_left'][0], i['top_left'][1]), (i['bottom_right'][0], i['bottom_right'][1]), color, thickness)
            if 'association' in i.keys():
                if 'cluster' in i.keys() and i['cluster']:
                    cv2.rectangle(image, (i['top_left'][0], i['top_left'][1]), (i['bottom_right'][0], i['bottom_right'][1]), grouped_association_color, grouped_association_thickness)
                    cv2.putText(image, i['association'], (i['top_right'][0], i['top_right'][1]), font, font_scale, grouped_association_color, font_thickness, cv2.LINE_AA)
                    continue
                # cv2.rectangle(image, (i['top_left'][0], i['top_left'][1]), (i['bottom_right'][0], i['bottom_right'][1]), association_color, association_thickness)
                # cv2.putText(image, i['association']+'  '+i['text'], (i['top_right'][0], i['top_right'][1]), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        except Exception as e:
            print(e)
    return image

def process_doc(image=None, json_path=None, verbose=True, write_image=False, key_fields=None, form_threshold=0.5, res_out=None, duplicate_tokens=None, key_threshold=None):
    assert key_fields is not None, 'key_fields field cannot be none' 
    data = None
    image_path = image
    image = cv2.imread(image)[:, :, ::-1]
    with open(json_path, 'r') as f:
        data = json.load(f)
    detections = data['annotations']
    original_detections = copy.deepcopy(detections)
    key_fields_detections_clusterd = get_key_fields(detections, image.shape, key_fields, duplicate_tokens)
    clustured_detections = get_associated_fields(key_fields_detections_clusterd, detections, image.shape[0], image.shape[1], key_fields, form_threshold, key_threshold)
    if verbose:
        image_actual = view_bbox_true(image, original_detections)
        image_key_clusterd = view_bbox_true(image, key_fields_detections_clusterd)
        image_key_clusterd = view_bbox_true(image_key_clusterd, clustured_detections, printing_association=True)
        cv2.imshow("Image - Actual", image_actual)
        cv2.imshow("Image - Key clusterd", image_key_clusterd)
        cv2.waitKey(0)
        if write_image:
            cv2.imwrite(os.path.join(res_out, image_path.split('/')[-1]), image_key_clusterd[:, :, ::-1])
        cv2.destroyAllWindows()
    return clustured_detections

def res_dict(doc_type, res, image_path):
    image_name = image_path.split('/')[-1]
    file_name = image_name.split('.')[0]
    for i in res:
        if 'cluster' in i.keys() and i['cluster']:
            i['association'] = re.sub(r'[\s:,]', ' ', i['association'].replace('/', '_').strip().lower().replace('  ', ' ').replace(' ', '_').replace('.', '_').replace('__', '_'))
    cluster = [i for i in res if 'cluster' in i.keys()]
    res_dict = {
        i: '' for i in conf.DF_COLS[doc_type]
    }
    res_dict['gcs_link'], res_dict['file_name'] = '', ''
    for i in cluster:
        if i['association'] in res_dict.keys() or i['association'] in conf.SPECIAL_TOKEN:
            if i['association'] == 'tempat_tgl_lahir'or ((i['association'] == 'tempat' or i['association'] == 'lahir') and doc_type == 'ktp'):
                tempat_tgl_lahir = i['text'].split(' ')
                if len(tempat_tgl_lahir) == 1:
                    tempat, tgl_lahir = tempat_tgl_lahir[0], tempat_tgl_lahir[0]
                else:
                    tempat, tgl_lahir = ' '.join(i['text'].split(' ')[:-1]), i['text'].split(' ')[-1]
                res_dict['tempat'] = tempat
                res_dict['tgl_lahir'] = tgl_lahir
            elif i['association'] == 'berlaku':
                res_dict['berlaku_hingga'] = i['text']
            else:
                res_dict[i['association']] = i['text']
        res_dict['gcs_link'] = conf.GCS_BUCKET[doc_type] + image_name
        res_dict['file_name'] = file_name
    print(res_dict)
    return res_dict

def main(args=None):
    record = []
    df = pd.read_csv(args.csv_file)
    images = os.listdir(args.input_directory)
    random.shuffle(images)
    for _, image_list in enumerate(images):
        # if image_list == "STNK9.jpeg":
        if True:
            print(f"{_}. Processing {image_list} ...")
            image = os.path.join(args.input_directory, image_list)
            json_path = df[df['file_name'] == image.split('/')[-1].split('.')[0]]['json_path'].tolist()
            if len(json_path) > 1:
                print('Ambiguous file name')
            if args.sim:
                res = res_dict("sim", process_doc(image, json_path[0], args.view_output, write_image=args.write_image, key_fields=conf.SIM_KEY_FIELDS, form_threshold=conf.SIM_FORM_FIELD_THRESHOLD, res_out=args.image_output_dir), image)
                record.append(res)
            elif args.ktp:
                res = res_dict("ktp", process_doc(image, json_path[0], args.view_output, write_image=args.write_image, key_fields=conf.KTP_KEY_FIELDS, form_threshold=conf.KTP_FORM_FIELD_THRESHOLD, res_out=args.image_output_dir), image)
                record.append(res)
            elif args.stnk:
                res = res_dict("stnk", process_doc(image, json_path[0], args.view_output, write_image=args.write_image, key_fields=conf.STNK_KEY_FIELDS, form_threshold=conf.STNK_FORM_FIELD_THRESHOLD, res_out=args.image_output_dir, duplicate_tokens=conf.DUPLICATE_TOKEN, key_threshold=conf.STNK_KEY_THRESHOLD), image)
                record.append(res)
            break
    csv_file_path = '/'.join(args.csv_file.split('/')[:-1])
    # pd.DataFrame.from_records(record).to_csv(csv_file_path + '/' + 'sim_inference_expert_system_1.csv')

if __name__ == '__main__':
    # Fetching arguments.
    parser = argparse.ArgumentParser(description='Remove duplicate files.')
    parser.add_argument('-f', '--csv_file', type=str, default=conf.INTERIM_2021_06_14_ocr_kyc_pdfs_annotations_csv, metavar="\b", help='Path to csv file')
    parser.add_argument('-i', '--input_directory', type=str, default=conf.INTERIM_2021_06_14_ocr_kyc_pdfs_manual_dl, metavar="\b", help='Path to a document type directory')
    parser.add_argument('-n', '--num_images', type=int, default=1, metavar="\b", help='Number of images to process')
    parser.add_argument('-g', '--gcp_api', type=str2bool, default='y', metavar="\b", help='Use GCP Vision API to get OCR results')
    parser.add_argument('-v', '--view_output', type=str2bool, default='y', metavar="\b", help='View output?')
    parser.add_argument('-w', '--write_image', type=str2bool, default='n', metavar="\b", help='write image to current working dir?')
    parser.add_argument('-o', '--image_output_dir', type=str, default=conf.DATA_DIR_INTERIM, metavar="\b", help='Path to a document result output directory')
    parser.add_argument('-sim', '--sim', type=str2bool, default='y', metavar="\b", help='Process SIM documents?')
    parser.add_argument('-ktp', '--ktp', type=str2bool, default='y', metavar="\b", help='Process KTP documents?')
    parser.add_argument('-stnk', '--stnk', type=str2bool, default='y', metavar="\b", help='Process STNK documents?')
    args = parser.parse_args()
    main(args=args)