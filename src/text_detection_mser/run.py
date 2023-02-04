import os
import cv2
import argparse


OUTPUT_FOLDER = 'mser'

def str2bool(v):
    return v.lower() in ("yes", "Yes", "YES", "y", "true", "True", "TRUE", "t", "1")

def mk_dir(path=None):
    if not os.path.isdir(path):
        os.mkdir(path)
        os.mkdir(path+f'/{OUTPUT_FOLDER}')
    if not os.path.isdir(path+f'/{OUTPUT_FOLDER}'):
        os.mkdir(path+f'/{OUTPUT_FOLDER}')

def get_images(dir=None):
    image_items = [os.path.join(dir, i) for i in os.listdir(dir)] 
    return image_items, len(image_items)

def detect_text(image_path=None, ouput_directory=None):
    mser = cv2.MSER_create()
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(img, hulls, 1, (0, 255, 0))
    cv2.imwrite(os.path.join(ouput_directory, 'mser_' + image_path.split('/')[-1]), img)

def main(args=None):
    mk_dir(args.output_directory)
    image_items, image_items_length = get_images(args.input_directory)
    for ind, image_path in enumerate(image_items):
        print(f"Image {ind+1}/{image_items_length}: {image_path}", end='\n')
        detect_text(image_path, args.output_directory + f'/{OUTPUT_FOLDER}')
        if args.demo:
            break

if __name__ == '__main__':
    # Fetching arguments.
    parser = argparse.ArgumentParser(description='Text Detection')
    parser.add_argument('-i', '--input_directory', type=str, default='../../data/interim/2021-06-14_ocr_kyc-pdfs', metavar="\b", help='Path to directory containing images')
    parser.add_argument('-o', '--output_directory', type=str, default='../../data/output', metavar="\b", help='Path to the directory in which output will be saved')
    parser.add_argument('-d', '--demo', default='y', type=str2bool, metavar="\b", help='Try out with one image only')
    args = parser.parse_args()
    main(args)