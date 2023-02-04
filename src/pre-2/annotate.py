import os
import glob
import json
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from cloud_vision_api import *
from config import config as conf


load_dotenv()

version = '1.0'
source = 'Google Vision API'

def mk_dir(path=None):
    if not os.path.isdir(path):
        os.mkdir(path)

def filter_fun(x, ignore_folders):
    for i in ignore_folders:
        if f'/{i}/' in x:
            return False
    return True

def annotate_file(image=None, image_path=None, output_file_path=None, process_image_needed=False, image_output_directory=None):
    try:
        ocr_response = get_bboxes(image, verbose=False) if not process_image_needed else process_image(image_path, image_output_directory)
        data = {
            'version': version,
            'source': source,
            'raw': ocr_response[0],
            'annotations': []
        }
        for i in ocr_response[1]:
            data['annotations'].append(
                {
                    'text': i['label'],
                    'top_left': (i['x1'], i['y1']),
                    'top_right': (i['x2'], i['y2']),
                    'bottom_right': (i['x3'], i['y3']),
                    'bottom_left': (i['x4'], i['y4']),
                    'width': i['w'],
                    'height': i['h']
                }
            )
        with open(output_file_path, "w") as f: 
            json.dump(data, f)
        return True
    except Exception as e:
        print(e)
        return False

def process_image(image_path, image_output_directory):
    morph_kernel = np.ones((5, 5))
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
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
    # p1 = np.float32([[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]])
    # p2 = np.float32([[0, 0], [img_dup.shape[1], 0], [0, img_dup.shape[0]], [img_dup.shape[1], img_dup.shape[0]]])
    # mtx = cv2.getPerspectiveTransform(p1, p2)
    # res = cv2.warpPerspective(img_dup, mtx, (img_dup.shape[1], img_dup.shape[0]))
    # img = cv2.resize(img, dsize=(max(img.shape[0], 1080), max(img.shape[1], 720)), interpolation=cv2.INTER_LANCZOS4)
    image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=sharpen_kernel)
    cv2.imwrite(os.path.join(image_output_directory, image_path.split('/')[-1]), image_sharp)
    image_pil = Image.fromarray(image_sharp)
    return get_bboxes(image_pil, format='JPEG', verbose=False)

def main(args=None):
    df_dict = {
        'image_path': [],
        'json_path': [],
        'file_name': []
    }
    image_list = list(glob.glob(args.input_directory + '/*.jpg', recursive=True)) + list(glob.glob(args.input_directory + '/*.jpeg', recursive=True)) + list(glob.glob(args.input_directory + '/*.webp', recursive=True)) + list(glob.glob(args.input_directory + '/*.png', recursive=True))
    output_directory = args.input_directory if args.output_directory == '' else args.output_directory
    mk_dir(output_directory)
    for image_path in tqdm(image_list):
        try:
            image = Image.open(image_path)
            json_path = os.path.join(output_directory, image_path.split('/')[-1].split('.')[0] + '.json')
            annotate_file(image, image_path, json_path, args.process_image, args.image_output_directory)
            df_dict['image_path'].append(image_path)
            df_dict['json_path'].append(json_path)
            df_dict['file_name'].append(image_path.split('/')[-1].split('.')[0])
        except Exception as e:
            print(e)
            continue
    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(output_directory, 'sim_ocr_annotations.csv'))


if __name__ == '__main__':
    # Fetching arguments.
    parser = argparse.ArgumentParser(description='Annotate images')
    parser.add_argument('-i', '--input_directory', type=str, default=conf.DATA_DIR_INTERIM, metavar="\b", help='Path to directory from which images will be annotated')
    parser.add_argument('-o', '--output_directory', type=str, default=conf.INTERIM_2021_06_14_ocr_kyc_pdfs_annotations, metavar="\b", help='Path to directory in which annotations will be stored')
    parser.add_argument('-io', '--image_output_directory', type=str, default=conf.INTERIM_2021_06_14_ocr_kyc_pdfs_annotations, metavar="\b", help='Path to image directory in which processed image will be stored')
    parser.add_argument('-ig','--ignore_folders', nargs='+', default=['2021-06-14_ocr_kyc-pdfs_annotations', '2021-06-14_ocr_kyc-pdfs_manual', '2021-08-10_text_recognition'], help='List of folders to be ignored')
    parser.add_argument('-p', '--process_image', type=str2bool, default='n', metavar="\b", help='Process image')
    args = parser.parse_args()
    main(args=args)