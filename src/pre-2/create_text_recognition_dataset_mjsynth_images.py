import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from config import config as conf


def mk_dir(path=None):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)

def extract(input_path, output_path):
    df_dict = {
        'folder_name': [],
        'image_path': [],
        'image_file_name': [],
        'label': [], 
        'parent_image_path': [],
        'parent_json_path': []
    }
    mk_dir(output_path)
    files = os.listdir(input_path)
    for item in tqdm(files):
        try:
            shutil.copy2(os.path.join(input_path, item), os.path.join(output_path, item))
            df_dict['folder_name'].append(output_path.split('/')[-1])
            df_dict['image_path'].append(os.path.join(output_path, item))
            df_dict['image_file_name'].append(item)
            label = item.split('_')[1]
            df_dict['label'].append(label.lower())
            df_dict['parent_image_path'].append(pd.NA)
            df_dict['parent_json_path'].append(pd.NA)
        except Exception as e:
            print(e)
            continue
    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(output_path, 'text_recognition_mjsynth_annots.csv'))

def main(args=None):
    extract(args.input_directory, args.output_directory)

if __name__ == '__main__':
    # Fetching arguments.
    parser = argparse.ArgumentParser(description='Annotate images')
    parser.add_argument('-i', '--input_directory', type=str, default=conf.RAW_mjsynth, metavar="\b", help='Input directory containing data')
    parser.add_argument('-o', '--output_directory', type=str, default=conf.INTERIM_2021_08_17_text_recognition_mjsynth_images, metavar="\b", help='Output directory to store data')
    args = parser.parse_args()
    main(args=args)