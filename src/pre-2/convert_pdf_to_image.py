import os
import argparse
from tqdm import tqdm
from pdf2image import convert_from_path
from config import config as conf
 

def pdf_to_image(folder=None, output_directory=None):
    pdfs = os.listdir(folder)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    for pdf in tqdm(pdfs):
        images = convert_from_path(os.path.join(folder, pdf))
        for i in range(len(images)):
            image_name = os.path.join(output_directory, pdf.split('.')[0] + '_' + str(i) + '.jpg')
            images[i].save(image_name.replace(' ', '_'), 'JPEG')

def main(args=None):
    pdf_to_image(args.input_dir, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PDF to image converter.')
    parser.add_argument('-i', '--input_dir', type=str, default=conf.RAW_2021_06_14_ocr_kyc_pdfs, metavar='\b', help='Path to PDFs directory')
    parser.add_argument('-o', '--output_dir', type=str, default=conf.INTERIM_PDF_TO_IMAGES_2021_06_14_ocr_kyc_pdfs_images, metavar='\b',help='Path to output directory')
    args = parser.parse_args()
    main(args=args)