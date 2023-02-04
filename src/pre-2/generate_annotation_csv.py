import copy
import argparse
import pandas as pd


def main(args=None):
   df_gt = pd.read_csv(args.input_csv)
   columns = ['id', 'image_id', 'internal_image_id', 'url', 'image'] + [i for i in df_gt.columns if '_annotated' in i]
   cleaned_columns = [i.replace("_annotated", '') for i in columns]
   df_annot = copy.deepcopy(df_gt[columns])
   df_annot.fillna('', inplace=True)
   df_annot['image'] = df_annot['image'].apply(lambda x: x.split('?')[0].split('/')[-1])
   df_annot.columns = cleaned_columns
   df_annot.to_csv(args.output_csv)


if __name__ == '__main__':
    # Fetching arguments.
    parser = argparse.ArgumentParser(description='Annotate images')
    parser.add_argument('-i', '--input_csv', type=str, metavar="\b", help='Path to input csv')
    parser.add_argument('-o', '--output_csv', type=str, metavar="\b", help='Path to output csv')
    args = parser.parse_args()
    main(args=args)