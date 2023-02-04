import io
import os
import json
import argparse
from PIL import Image
from dotenv import load_dotenv
from google.cloud import documentai_v1 as documentai
from google.protobuf.json_format import MessageToJson
from config import config as conf


load_dotenv()

def mk_dir(dir=None):
    if not os.path.isdir(dir):
        os.mkdir(dir)

def image_to_byte_array(image: Image, format: str = None):
    imgByteArr = io.BytesIO()
    if not format:
        image.save(imgByteArr, format=image.format)
    else:
        image.save(imgByteArr, format=format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

def get_key_value_form_fields(document_text, form_fields, item):
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
        key_x2, key_y2 = float(key['boundingPoly']['normalizedVertices'][1].get('x', 0)), float(key['boundingPoly']['normalizedVertices'][0].get('y', 0))
        key_x3, key_y3 = float(key['boundingPoly']['normalizedVertices'][2].get('x', 0)), float(key['boundingPoly']['normalizedVertices'][0].get('y', 0))
        key_x4, key_y4 = float(key['boundingPoly']['normalizedVertices'][3].get('x', 0)), float(key['boundingPoly']['normalizedVertices'][0].get('y', 0))
        key_confidence = float(key['confidence'])
        value_string = ''
        try:
            for value_segment in value['textAnchor']['textSegments']:
                value_string += document_text[int(value_segment.get('startIndex', 0)): int(value_segment.get('endIndex', 0))]
        except Exception as e:
            print(f'Error occurred with {item} at value textAnchor extraction [form field] - {e}')
        value_x1, value_y1 = float(value['boundingPoly']['normalizedVertices'][0].get('x', 0)), float(value['boundingPoly']['normalizedVertices'][0].get('y', 0))
        value_x2, value_y2 = float(value['boundingPoly']['normalizedVertices'][1].get('x', 0)), float(value['boundingPoly']['normalizedVertices'][0].get('y', 0))
        value_x3, value_y3 = float(value['boundingPoly']['normalizedVertices'][2].get('x', 0)), float(value['boundingPoly']['normalizedVertices'][0].get('y', 0))
        value_x4, value_y4 = float(value['boundingPoly']['normalizedVertices'][3].get('x', 0)), float(value['boundingPoly']['normalizedVertices'][0].get('y', 0))
        value_confidence = float(value['confidence'])
        form_dict['key'] = {
            'text': key_string,
            'bbox': {
                'x1': key_x1,
                'y1': key_y1,
                'x2': key_x2,
                'y2': key_y2,
                'x3': key_x3,
                'y3': key_y3,
                'x4': key_x4,
                'y4': key_y4
            },
            'confidence': key_confidence
        }
        form_dict['value'] = {
            'text': value_string,
            'bbox': {
                'x1': value_x1,
                'y1': value_y1,
                'x2': value_x2,
                'y2': value_y2,
                'x3': value_x3,
                'y3': value_y3,
                'x4': value_x4,
                'y4': value_y4
            },
            'confidence': value_confidence
        }
        form_list.append(form_dict)
    return form_list

def get_tokens(document_text, tokens):
    token_list = []
    for token in tokens:
        string = ''
        for segment in token['layout']['textAnchor']['textSegments']:
            string += document_text[int(segment.get('startIndex', 0)): int(segment.get('endIndex', 0))]
        x1, y1 = float(token['layout']['boundingPoly']['vertices'][0].get('x', 0)), float(token['layout']['boundingPoly']['vertices'][0].get('y', 0))
        x2, y2 = float(token['layout']['boundingPoly']['vertices'][1].get('x', 0)), float(token['layout']['boundingPoly']['vertices'][0].get('y', 0))
        x3, y3 = float(token['layout']['boundingPoly']['vertices'][2].get('x', 0)), float(token['layout']['boundingPoly']['vertices'][0].get('y', 0))
        x4, y4 = float(token['layout']['boundingPoly']['vertices'][3].get('x', 0)), float(token['layout']['boundingPoly']['vertices'][0].get('y', 0))
        n_x1, n_y1 = float(token['layout']['boundingPoly']['normalizedVertices'][0].get('x', 0)), float(token['layout']['boundingPoly']['normalizedVertices'][0].get('y', 0))
        n_x2, n_y2 = float(token['layout']['boundingPoly']['normalizedVertices'][1].get('x', 0)), float(token['layout']['boundingPoly']['normalizedVertices'][0].get('y', 0))
        n_x3, n_y3 = float(token['layout']['boundingPoly']['normalizedVertices'][2].get('x', 0)), float(token['layout']['boundingPoly']['normalizedVertices'][0].get('y', 0))
        n_x4, n_y4 = float(token['layout']['boundingPoly']['normalizedVertices'][3].get('x', 0)), float(token['layout']['boundingPoly']['normalizedVertices'][0].get('y', 0))
        confidence = float(token['layout']['confidence'])
        token_list.append({
            'text': string,
            'bbox': {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'x3': x3,
                'y3': y3,
                'x4': x4,
                'y4': y4
            },
            'normalized_bbox': {
                'x1': n_x1,
                'y1': n_y1,
                'x2': n_x2,
                'y2': n_y2,
                'x3': n_x3,
                'y3': n_y3,
                'x4': n_x4,
                'y4': n_y4
            },
            'confidence': confidence
        })
    return token_list

def document_ai(input_directory=None, ouput_directory=None):
    print(f'Input directory - {input_directory}\nOutput directory - {ouput_directory}')
    mk_dir(ouput_directory)
    client_options = {"api_endpoint": "{}-documentai.googleapis.com".format(os.getenv('GCP_PROCESSOR_LOCATION', 'eu'))}
    client = documentai.DocumentProcessorServiceClient(client_options=client_options)
    name = f"projects/{os.getenv('GCP_PROJECT_ID', None)}/locations/{os.getenv('GCP_PROCESSOR_LOCATION', 'eu')}/processors/{os.getenv('GCP_PROCESSOR_ID', None)}"
    for ind, item in enumerate([_ for _ in os.listdir(input_directory) if '.jpg' in _.lower() or '.png' in _.lower() or '.jpeg' in _.lower()]):
        file_name = item.split('.')[0]
        if os.path.exists(os.path.join(ouput_directory, file_name + '_document_ai' + '.json')):
            print(f'{ind+1}. Skipping {item} ...')
            continue
        print(f'{ind+1}. Processing {item} ...')
        image = Image.open(os.path.join(input_directory, item))
        image_bytes = image_to_byte_array(image)
        document = {"content": image_bytes, "mime_type": "image/jpeg"}
        request = {"name": name, "raw_document": document}
        result = client.process_document(request=request)
        result = json.loads(MessageToJson(result._pb))
        res = {}
        form_fields, tokens, image_width, image_height = [], [], [], []
        for _, page in enumerate(result['document']['pages']):
            try:
                form_fields.append(get_key_value_form_fields(result['document']['text'], page['formFields'], item))
                tokens.append(get_tokens(result['document']['text'], page['tokens']))
                image_width.append(page['dimension']['width'])
                image_height.append(page['dimension']['height'])
            except Exception as e:
                print(f'Error occurred with {item} at page extraction [form field] - {e}')
                continue
        res['form_field'] = form_fields
        res['tokens'] = tokens
        res['image_width'] = image_width
        res['image_height'] = image_height
        res['raw'] = result
        with open(os.path.join(ouput_directory, file_name + '_document_ai' + '.json'), "w") as f: 
            json.dump(res, f)

def main(args=None):
    document_ai(args.input_directory, args.output_directory)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Google Document AI - Information extraction (Pass the image in JPG/JPEG format).')
    parser.add_argument('-i', '--input_directory', type=str, default=conf.INTERIM_2022_01_06_combined_orc_kyc + '/KTP', metavar="\b", help='Input directory')
    parser.add_argument('-o', '--output_directory', type=str, default=conf.INTERIM_2022_01_06_combined_orc_kyc + '/KTP', metavar="\b", help='Output directory')
    args = parser.parse_args()
    main(args=args)