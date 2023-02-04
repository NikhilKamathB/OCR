from ktpexpert import *
import logging
from tqdm import tqdm 
import argparse 
import os
import re
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from bpemb import BPEmb
from sklearn import svm
from sklearn import metrics
from pathlib import PosixPath
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import warnings
from pathlib import Path 
import glob
import math 
import cv2
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import numpy as np
import re
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import pickle
from functools import partial
from collections import defaultdict
from strsimpy.levenshtein import Levenshtein


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("csv_annotated")
    parser.add_argument("csv_inference")
    
    args = parser.parse_args()


    DATA_DIR = Path(args.data_dir)

    res = []

    df = pd.read_csv(DATA_DIR/args.csv_annotated)
    for _,row in tqdm(df.iterrows(), total=df.shape[0]):
        if row['is_invalid']:
            continue 
        try:
            file = row['image']
            FILE_TYPE = row['file_type']
            base = os.path.basename(file).split('.')[0]
            yolo_file = DATA_DIR/FILE_TYPE / (base + ".txt")
            json_file = DATA_DIR/FILE_TYPE/ (base + ".json")
            img_file = DATA_DIR/FILE_TYPE/ file 
            
            df = process_ktp(json_file)

            # plot_figure(str(img_file), df, show=True, waitkey=0, out_file=str(Path("/tmp/ocr/ktp") / file))
            
            rr = row.to_dict()
            rr.update(to_json(df))
            
            res.append(rr)
            

        except Exception as e:
            logging.error("exception while processing %(row)s", {"row": row})
            continue

    res = pd.DataFrame(res)
    print(res.shape)    
    res.to_csv(args.csv_inference, index=False)