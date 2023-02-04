from operator import index
import os
import io
import re
import cv2
import json
import copy
import random
import argparse
from PIL import Image
from matplotlib import image
import numpy as np
import pandas as pd
from sympy import arg
from tomlkit import key
from ner import *
from config import config as conf
from strsimpy.levenshtein import Levenshtein
from google.cloud import documentai_v1 as documentai
from google.protobuf.json_format import MessageToJson


def str2bool(v):
    return v.lower() in ("yes", "Yes", "YES", "y", "true", "True", "TRUE", "t", "1")

def view_bbox_true(image, detections, color=(0, 0, 255), association_color=(255, 0, 0), grouped_association_color=(255, 0, 0), thickness=1, association_thickness=1, grouped_association_thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_color=(255, 0, 0), associated_font_color=(0, 255, 0), font_thickness=1, printing_association=False):
    image = np.array(image)
    for i in detections:
        try:
            if not printing_association:
                cv2.rectangle(image, (i['top_left'][0], i['top_left'][1]), (i['bottom_right'][0], i['bottom_right'][1]), color, thickness)
            if 'association' in i.keys():
                if 'cluster' in i.keys() and i['cluster']:
                    cv2.rectangle(image, (i['top_left'][0], i['top_left'][1]), (i['bottom_right'][0], i['bottom_right'][1]), grouped_association_color, grouped_association_thickness)
                    cv2.putText(image, i['association'], (i['top_right'][0], i['top_right'][1]), font, font_scale, grouped_association_color, font_thickness, cv2.LINE_AA)
        except Exception as e:
            print(e)
    return image

def image_to_byte_array(image: Image, format: str = None):
    imgByteArr = io.BytesIO()
    if not format:
        image.save(imgByteArr, format=image.format)
    else:
        image.save(imgByteArr, format=format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

def get_key_value_form_fields(document_text, form_fields, item, width, height):
    form_list = []
    for form_field in form_fields:
        form_dict = {}
        key, value = form_field['fieldName'], form_field['fieldValue']
        key_string = ''
        try:
            for key_segment in key['textAnchor']['textSegments']:
                key_string += document_text[int(key_segment.get('startIndex', 0)): int(key_segment.get('endIndex', 0))]
        except Exception as e:
            print(f'Error occurred with {item} at key textAnchor extraction [form field] - {e}')
        key_x1, key_y1 = float(key['boundingPoly']['normalizedVertices'][0].get('x', 0)), float(key['boundingPoly']['normalizedVertices'][0].get('y', 0))
        key_x2, key_y2 = float(key['boundingPoly']['normalizedVertices'][1].get('x', 0)), float(key['boundingPoly']['normalizedVertices'][1].get('y', 0))
        key_x3, key_y3 = float(key['boundingPoly']['normalizedVertices'][2].get('x', 0)), float(key['boundingPoly']['normalizedVertices'][2].get('y', 0))
        key_x4, key_y4 = float(key['boundingPoly']['normalizedVertices'][3].get('x', 0)), float(key['boundingPoly']['normalizedVertices'][3].get('y', 0))
        key_confidence = float(key['confidence'])
        value_string = ''
        try:
            for value_segment in value['textAnchor']['textSegments']:
                value_string += document_text[int(value_segment.get('startIndex', 0)): int(value_segment.get('endIndex', 0))]
        except Exception as e:
            print(f'Error occurred with {item} at value textAnchor extraction [form field] - {e}')
        value_x1, value_y1 = float(value['boundingPoly']['normalizedVertices'][0].get('x', 0)), float(value['boundingPoly']['normalizedVertices'][0].get('y', 0))
        value_x2, value_y2 = float(value['boundingPoly']['normalizedVertices'][1].get('x', 0)), float(value['boundingPoly']['normalizedVertices'][1].get('y', 0))
        value_x3, value_y3 = float(value['boundingPoly']['normalizedVertices'][2].get('x', 0)), float(value['boundingPoly']['normalizedVertices'][2].get('y', 0))
        value_x4, value_y4 = float(value['boundingPoly']['normalizedVertices'][3].get('x', 0)), float(value['boundingPoly']['normalizedVertices'][3].get('y', 0))
        value_confidence = float(value['confidence'])
        form_dict['key'] = {
            'text': key_string,
            'bbox': {
                'x1': key_x1*width,
                'y1': key_y1*height,
                'x2': key_x2*width,
                'y2': key_y2*height,
                'x3': key_x3*width,
                'y3': key_y3*height,
                'x4': key_x4*width,
                'y4': key_y4*height
            },
            'confidence': key_confidence
        }
        form_dict['value'] = {
            'text': value_string,
            'bbox': {
                'x1': value_x1*width,
                'y1': value_y1*height,
                'x2': value_x2*width,
                'y2': value_y2*height,
                'x3': value_x3*width,
                'y3': value_y3*height,
                'x4': value_x4*width,
                'y4': value_y4*height
            },
            'confidence': value_confidence
        }
        form_list.append(form_dict)
    return form_list

def get_keys_visualization(res):
    keys = []
    for r in res:
        r['key']['top_left'] = [int(r['key']['bbox']['x1']), int(r['key']['bbox']['y1'])]
        r['key']['top_right'] = [int(r['key']['bbox']['x2']), int(r['key']['bbox']['y2'])]
        r['key']['bottom_right'] = [int(r['key']['bbox']['x3']), int(r['key']['bbox']['y3'])]
        r['key']['bottom_left'] = [int(r['key']['bbox']['x4']), int(r['key']['bbox']['y4'])]
        keys.append(r['key'])
    return keys

def clean_result(res, keys):
    res_dict = {}
    values = []
    for key in keys:
        for r in res:
            if Levenshtein().distance(key.lower().strip(), r['key']['text'].strip().lower()) < get_threshold(len(key)):
                r['value']['top_left'] = [int(r['value']['bbox']['x1']), int(r['value']['bbox']['y1'])]
                r['value']['top_right'] = [int(r['value']['bbox']['x2']), int(r['value']['bbox']['y2'])]
                r['value']['bottom_right'] = [int(r['value']['bbox']['x3']), int(r['value']['bbox']['y3'])]
                r['value']['bottom_left'] = [int(r['value']['bbox']['x4']), int(r['value']['bbox']['y4'])]
                r['value']['association'] = r['key']['text'].strip().lower()
                r['value']['cluster'] = True
                values.append(r['value'])
                res_dict[key] = r['value']['text'].strip().lower()
                break
    return res_dict, values

def document_ai(image=None, keys=None, verbose=True):
    res = {}
    image_path = image
    mime_type = conf.MIME_TYPES[image_path.split('.')[-1].lower()]
    client_options = {"api_endpoint": "{}-documentai.googleapis.com".format(os.getenv('GCP_PROCESSOR_LOCATION', 'us'))}
    client = documentai.DocumentProcessorServiceClient(client_options=client_options)
    name = f"projects/{os.getenv('GCP_PROJECT_ID', None)}/locations/{os.getenv('GCP_PROCESSOR_LOCATION', 'us')}/processors/{os.getenv('GCP_PROCESSOR_ID', None)}"
    image = Image.open(image_path)
    image_bytes = image_to_byte_array(image)
    document = {"content": image_bytes, "mime_type": f"image/{mime_type}"}
    request = {"name": name, "raw_document": document}
    result = json.loads(MessageToJson(client.process_document(request=request)._pb))
    form_fields = []
    for _, page in enumerate(result['document']['pages']):
        try:
            form_fields.append(get_key_value_form_fields(result['document']['text'], page['formFields'], image_path, page['dimension']['width'], page['dimension']['height']))
        except Exception as e:
            print(f'Error occurred with {image_path} at page extraction [form field] - {e}')
            continue
    res['form_field'] = form_fields
    detected_keys = get_keys_visualization(res['form_field'][0])
    res_dict, detected_values = clean_result(res['form_field'][0], keys)
    print(res_dict)
    if verbose:
        image = cv2.imread(image_path)[:, :, ::-1]
        image_key_clusterd = view_bbox_true(image, detected_keys)
        image_key_clusterd = view_bbox_true(image_key_clusterd, detected_values, printing_association=True)
        cv2.imshow("Image - Key clusterd", image_key_clusterd)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return res_dict

def process_doc(image=None, json_path=None, verbose=True, write_image=False, key_fields=None, key_field_lines=None, form_threshold=0.5, res_out=None):
    assert key_fields is not None, 'key_fields field cannot be none' 
    data = None
    image_path = image
    image = cv2.imread(image)[:, :, ::-1]
    image_org = copy.deepcopy(image)
    with open(json_path, 'r') as f:
        data = json.load(f)
    detections = data['annotations']
    original_detections = copy.deepcopy(detections)
    key_fields_detections_clusterd = get_key_fields(detections, image.shape, key_fields)
    clustured_detections = get_associated_fields(key_fields_detections_clusterd, detections, image.shape[0], image.shape[1], key_fields, key_field_lines, form_threshold)
    if verbose:
        image_actual = view_bbox_true(image, original_detections)
        image_key_clusterd = view_bbox_true(image, key_fields_detections_clusterd)
        image_key_clusterd = view_bbox_true(image_key_clusterd, clustured_detections, printing_association=True)
        cv2.imshow("Image - Original", image_org)
        cv2.imshow("Image - OCR engine output", image_actual)
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
    res_dict['gcs_link'], res_dict['file_name'], res_dict['image'] = '', '', ''
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
            elif i['association'] == 'berlaku' or i['association'] == 'berlaku_sim' or i['association'] == 'hingga':
                res_dict['berlaku_hingga'] = i['text'].replace('s d', '').strip()
            elif i['association'] == 'status':
                res_dict['status_perkawinan'] = i['text'].strip()
            else:
                res_dict[i['association']] = i['text']
        res_dict['gcs_link'] = conf.GCS_BUCKET[doc_type] + image_name
        res_dict['file_name'] = file_name
        res_dict['image'] = image_name.replace(',', '').replace(' ', '_').replace('__', '_')
    return res_dict

def main(args=None):
    record = []
    doc_type = None
    df = pd.concat([pd.read_csv(i) for i in args.csv_file])
    images = [os.path.join(dir, img) for dir in args.input_directory for img in os.listdir(dir)]
    random.shuffle(images)
    for ind, image in enumerate(images):
        if 'Copy_STNK_&_KTP_0c' in image:
            file_name = image.split('/')[-1]
            processed_image_path = '/'.join(image.split('/')[:-2]) + '/images_processed/' + file_name
            if os.path.exists(processed_image_path):
                image_path = processed_image_path
            else:
                image_path = image
            print(f"{ind+1}. Processing {image_path} ...")
            json_path = df[df['file_name'] == image.split('/')[-1].split('.')[0]]['json_path'].tolist()
            if len(json_path) > 1:
                print('Ambiguous file name')
            json_file = json_path[0].split('/')[-1]
            json_path = os.path.join('/'.join(image.split('/')[:-2]), 'json', json_file)
            if args.sim:
                res = res_dict("sim", process_doc(image_path, json_path, args.view_output, write_image=args.write_image, key_fields=conf.SIM_KEY_FIELDS, key_field_lines=conf.SIM_KEY_FIELDS_LINES, form_threshold=conf.SIM_FORM_FIELD_THRESHOLD, res_out=args.image_output_dir), image_path)
                record.append(res)
                doc_type = "sim"
            elif args.ktp:
                res = res_dict("ktp", process_doc(image_path, json_path, args.view_output, write_image=args.write_image, key_fields=conf.KTP_KEY_FIELDS, key_field_lines=conf.SIM_KEY_FIELDS_LINES, form_threshold=conf.KTP_FORM_FIELD_THRESHOLD, res_out=args.image_output_dir), image_path)
                record.append(res)
                doc_type = "ktp"
            # elif args.stnk:
            #     res = res_dict("stnk", process_doc(image_path, json_path, args.view_output, write_image=args.write_image, key_fields=conf.STNK_KEY_FIELDS, form_threshold=conf.STNK_FORM_FIELD_THRESHOLD, res_out=args.image_output_dir), image_path)
            #     record.append(res)
            #     doc_type = "stnk"
            elif args.stnk_samsat and args.document_ai:
                document_ai(image=image_path, keys=conf.STNK_SAMSAT_KEY_FIELDS_DOC_AI, verbose=args.view_output)
                doc_type = "stnk_samsat_document_ai"
            # elif args.stnk_samsat:
            #     res = res_dict("stnk_samsat", process_doc(image_path, json_path, args.view_output, write_image=args.write_image, key_fields=conf.STNK_SAMSAT_KEY_FIELDS, form_threshold=conf.STNK_SAMSAT_FORM_FIELD_THRESHOLD, res_out=args.image_output_dir), image_path)
            #     record.append(res)
            #     doc_type = "stnk_samsat"
            if ind+1 == args.num_images:
                break
    if args.output_csv_file and doc_type:
        pd.DataFrame.from_records(record).to_csv(args.output_csv_file + '/' + f'{doc_type}_inference_expert_system_2.csv')

if __name__ == '__main__':
    # Fetching arguments.
    parser = argparse.ArgumentParser(description='Remove duplicate files.')
    parser.add_argument('-f', '--csv_file', nargs='+', default=[conf.INTERIM_2021_06_14_ocr_kyc_pdfs_annotations_csv], metavar="\b", help='Path to csv file')
    parser.add_argument('-of', '--output_csv_file', type=str, default=None, metavar="\b", help='Path to inference csv file')
    parser.add_argument('-i', '--input_directory', nargs='+', default=[conf.INTERIM_2021_06_14_ocr_kyc_pdfs_manual_dl], metavar="\b", help='Path to a document type directory')
    parser.add_argument('-n', '--num_images', type=int, default=0, metavar="\b", help='Number of images to process')
    parser.add_argument('-g', '--gcp_api', type=str2bool, default='y', metavar="\b", help='Use GCP Vision API to get OCR results')
    parser.add_argument('-v', '--view_output', type=str2bool, default='y', metavar="\b", help='View output?')
    parser.add_argument('-w', '--write_image', type=str2bool, default='n', metavar="\b", help='write image to current working dir?')
    parser.add_argument('-o', '--image_output_dir', type=str, default=conf.DATA_DIR_INTERIM, metavar="\b", help='Path to a document result output directory')
    parser.add_argument('-sim', '--sim', type=str2bool, default='n', metavar="\b", help='Process SIM documents?')
    parser.add_argument('-ktp', '--ktp', type=str2bool, default='n', metavar="\b", help='Process KTP documents?')
    parser.add_argument('-stnk', '--stnk', type=str2bool, default='n', metavar="\b", help='Process STNK documents?')
    parser.add_argument('-ss', '--stnk_samsat', type=str2bool, default='n', metavar="\b", help='Process STNK SAMSAT documents?')
    parser.add_argument('-d', '--document_ai', type=str2bool, default='n', metavar="\b", help='Process STNK SAMSAT documents using Document AI?')
    args = parser.parse_args()
    main(args=args)