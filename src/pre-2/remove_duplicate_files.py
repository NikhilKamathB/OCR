import os
import argparse
from tqdm import tqdm
import imagehash
from PIL import Image 
from config import config as conf


def remove_duplicates(folder=None):
    duplicates = []
    hash_keys = dict()
    for index, filename in tqdm(enumerate(os.listdir(folder))):
        if os.path.isfile(os.path.join(folder, filename)):
            with open(os.path.join(folder, filename), 'rb') as f:
                img = Image.open(f)
                filehash = imagehash.average_hash(img)
            if filehash not in hash_keys: 
                hash_keys[filehash] = index
            else:
                duplicates.append(os.path.join(folder, filename))
    if duplicates:
        print(f"Removing {len(duplicates)} duplicates")
        for file in duplicates:
            os.remove(file)
    else:
        print('No duplicates found')

def main(args=None):
    remove_duplicates(args.input_dir)

if __name__ == '__main__':
    # Fetching arguments.
    parser = argparse.ArgumentParser(description='Remove duplicate files.')
    parser.add_argument('-i', '--input_dir', type=str, default=conf.INTERIM_PDF_TO_IMAGES_2021_06_14_ocr_kyc_pdfs_images, metavar="\b", help='Path to directory from which duplicate files will be removed')
    args = parser.parse_args()
    main(args=args)