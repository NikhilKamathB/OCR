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


def bboxs_for_file(json_file):
    if not os.path.exists(json_file):
        print(f"{json_file} does not exist")
        print(os.getcwd())
        return []

    text_response = None 
    with open(json_file, "r") as of:
        text_response = json.load(of)

    imgw, imgh = text_response['fullTextAnnotation']['pages'][0]['width'], text_response['fullTextAnnotation']['pages'][0]['height']
    bboxs = []

    if ('textAnnotations' in text_response):
        for i,text in enumerate(text_response['textAnnotations']):
            boxes = {}
        
            x1 = text['boundingPoly']['vertices'][0].get('x',0)
            y1 = text['boundingPoly']['vertices'][0].get('y',0)
            x2 = text['boundingPoly']['vertices'][1].get('x',0)
            y2 = text['boundingPoly']['vertices'][1].get('y',0)
            x3 = text['boundingPoly']['vertices'][2].get('x',0)
            y3 = text['boundingPoly']['vertices'][2].get('y',0)
            x4 = text['boundingPoly']['vertices'][3].get('x',0)
            y4 = text['boundingPoly']['vertices'][3].get('y',0)
            

            xmin = min(x1, x4)
            xmax = max(x2, x3)
            cx = (xmax + xmin) // 2

            ymin = min(y1, y2)
            ymax = max(y3, y4)
            cy = (ymax + ymin) // 2

            center = ( cx/imgw , cy/imgh )
            width = (x2-x1)/imgw
            height = (y3-y2)/imgh

            bboxs.append((text['description'], *center, width, height, imgw, imgh, i==0))
    return bboxs

def load_data(files):
    """ load the ocr bboxes of given files into a dataframe. with file column as key."""
    rows = []

    bboxs_all = []
    for f in files: 
        bboxs = bboxs_for_file(f)
        bboxs_all.extend([(os.path.basename(f), *bb) for bb in bboxs])
        
    df = pd.DataFrame(bboxs_all, columns=['file', 'text', 'cx', 'cy', 'w', 'h', 'imgw', 'imgh', 'is_first'])
    df['cx'] = df['cx'].astype(np.float32)
    df['cy'] = df['cy'].astype(np.float32)
    df['w'] = df['w'].astype(np.float32)
    df['h'] = df['h'].astype(np.float32)
    df['is_first'] = df['is_first'].astype(bool)
    
    return df 

def process_features(df):

    df['f_isdigit'] = df.apply(lambda x: x['text'].isdigit(), axis=1)

    def is_date(x):
        from dateutil.parser import parse
        try: 
            parse(x['text'], fuzzy=False)
            return True
        except (ValueError, OverflowError):
            return False
        
    df['f_isdate'] = df.apply(is_date, axis=1)

    df['f_iszipcode'] = df.apply(lambda x: x['text'].isdigit() and len(x['text']) == 5, axis=1)
    df['f_isalpha'] = df.apply(lambda x: x['text'].isalpha(), axis=1)
    df['f_isalnum'] = df.apply(lambda x: x['text'].isalnum(), axis=1)
    df['f_isnum'] = df.apply(lambda x: x['text'].isnumeric(), axis=1)
    df['f_isnumsym'] = df.apply(lambda x: bool(re.search('^[0-9.,:;!?()]+$/', x['text'])), axis=1)
    df['f_isnumreal'] = df.apply(lambda x: x['text'].isdigit(), axis=1)
    # df['f_iscity'] = df.apply(lambda x: x['text'].isdigit(), axis=1)
    # df['f_iscurrency'] = df.apply(lambda x: x['text'].isdigit(), axis=1)
    # df['f_isgender'] = df.apply(lambda x: x['text'].isdigit(), axis=1)
    # df['f_isnationality'] = df.apply(lambda x: x['text'].isdigit(), axis=1)

    df['f_nik'] = df.apply(lambda x: (x['text'].upper() == 'NIK') or (x['text'].upper() == 'N.I.K.') or (x['text'].upper() == 'N.I.K'), axis=1)
    df['f_nama'] = df.apply(lambda x: x['text'].upper() == 'NAMA', axis=1)
    df['f_jenis'] = df.apply(lambda x: (x['text'].upper() == 'JENIS') or (x['text'].upper() == 'KELAMIN'), axis=1)
    df['f_tempat'] = df.apply(lambda x: (x['text'].upper() == 'TEMPAT') or (x['text'].upper() == 'TGL') or (x['text'].upper() == 'LAHIR'), axis=1)
    # df['f_tgllahir'] = df.apply(lambda x: (x['text'].upper() == 'TGL') or (x['text'].upper() == 'LAHIR'), axis=1)
    df['f_alamat'] = df.apply(lambda x: x['text'].upper() == 'ALAMAT', axis=1)
    df['f_rtrw'] = df.apply(lambda x: (x['text'].upper() == 'RT') or (x['text'].upper() == 'RW') or (x['text'].upper() == 'RT/RW') or (x['text'].upper() == 'RTRW') , axis=1)
    df['f_keldesa'] = df.apply(lambda x: (x['text'].upper() == 'KELDESA') or  (x['text'].upper() == 'KEL') or  (x['text'].upper() == 'DESA'), axis=1)
    df['f_kecamatan'] = df.apply(lambda x: x['text'].upper() == 'KECAMATAN', axis=1)
    df['f_agama'] = df.apply(lambda x: x['text'].upper() == 'AGAMA', axis=1)
    df['f_status_perkawinan'] = df.apply(lambda x: (x['text'].upper() == 'STATUS') or (x['text'].upper() == 'PERKAWINAN') , axis=1)
    df['f_pekerjaan'] = df.apply(lambda x: x['text'].upper() == 'PEKERJAAN', axis=1)
    df['f_kewarganegaraan'] = df.apply(lambda x: x['text'].upper() == 'KEWARGANEGARAAN', axis=1)
    df['f_berlaku'] = df.apply(lambda x: (x['text'].upper() == 'BERLAKU') or (x['text'].upper() == 'HINGGA'), axis=1)


        
    def expsys_ktp_key(x):

        ldist = Levenshtein()

        if (x['text'].upper() == 'NIK') or (x['text'].upper() == 'N.I.K.') or (x['text'].upper() == 'N.I.K'):
            return 1
        if x['text'].upper() == 'NAMA' or (int(ldist.distance('NAMA', x['text'].upper())) <= 1):
            return 2
        if (x['text'].upper() == 'JENIS') or (x['text'].upper() == 'KELAMIN') or (int(ldist.distance('KELAMIN', x['text'].upper())) <= 1):
            return 4
        if (x['text'].upper() == 'TEMPAT') or (x['text'].upper() == 'TGL') or (x['text'].upper() == 'LAHIR'):
            return 5
        if x['text'].upper() == 'ALAMAT' or (int(ldist.distance('ALAMAT', x['text'].upper())) <= 1):
            return 6
        if (x['text'].upper() == 'RT') or (x['text'].upper() == 'RW') or (x['text'].upper() == 'RT/RW') or (x['text'].upper() == 'RTRW') or (int(ldist.distance('RT/RW', x['text'].upper())) <= 1):
            return 7
        if (x['text'].upper() == 'KELDESA') or  (x['text'].upper() == 'KEL') or  (x['text'].upper() == 'DESA') or (x['text'].upper() == 'KEL/DESA') or (int(ldist.distance('KEL/DESA', x['text'].upper())) <= 1):
            return 8
        if (x['text'].upper() == 'KECAMATAN') or (x['text'].upper() == 'KECAMATAN:') or (int(ldist.distance('KECAMATAN', x['text'].upper())) <= 1):
            return 9
        if (x['text'].upper() == 'AGAMA'):
            return 10
        if (x['text'].upper() == 'STATUS') or (x['text'].upper() == 'PERKAWINAN') or  (int(ldist.distance('PERKAWINAN', x['text'].upper())) <= 1):
            return 11
        if x['text'].upper() == 'PEKERJAAN':
            return 12
        if (x['text'].upper() == 'KEWARGANEGARAAN') or (int(ldist.distance('KEWARGANEGARAAN', x['text'].upper())) <= 1):
            return 13
        if (x['text'].upper() == 'BERLAKU') or (x['text'].upper() == 'HINGGA') or (x['text'].upper() == 'HINGGA:') or (int(ldist.distance('BERLAKU', x['text'].upper())) <= 1) or (int(ldist.distance('HINGGA', x['text'].upper())) <= 1):
            return 14
        return 0

    df['f_y'] = df.apply(expsys_ktp_key, axis=1)

    df['f_len'] = df.apply(lambda x: len(x['text']), axis=1)

    df['f_issym'] = df.apply(lambda x: bool(re.search('^[.,:;!?()\/]+$', x['text'])), axis=1)



    # this assumes that is_first is always a first row. if we reindex the pandas dataframe,
    # then be sure to rewrite the ncx/ncy logic update.
    w = h = cx = cy = 0

    rows = defaultdict(list)

    for _, rr in df.iterrows():
        file = rr['file']
        if rr['is_first']:
            w = rr['w'] * rr['imgw'] 
            h = rr['h'] * rr['imgh']
            cx = rr['cx'] * rr['imgw']
            cy = rr['cy'] * rr['imgh']
            rows['ncx'].append(rr['cx'] * rr['imgw'] / w)
            rows['ncy'].append(rr['cy'] * rr['imgh'] / h)
            rows['nw'].append(rr['w'] * rr['imgw'] / w)
            rows['nh'].append(rr['h'] * rr['imgh'] / h)
        else:
            rows['ncx'].append(rr['cx'] * rr['imgw'] / w)
            rows['ncy'].append(rr['cy'] * rr['imgh'] / h)
            rows['nw'].append(rr['w'] * rr['imgw'] / w)
            rows['nh'].append(rr['h'] * rr['imgh'] / h)

    df['ncx'] = rows['ncx']
    df['ncy'] = rows['ncy']
    df['nw'] = rows['nw']
    df['nh'] = rows['nh']


    df['f_len'] = df.apply(lambda x: len(x['text']), axis=1).astype(int)
    df['cx'] = df['cx'].astype(np.float32)
    df['cy'] = df['cy'].astype(np.float32)
    df['w'] = df['w'].astype(np.float32)
    df['h'] = df['h'].astype(np.float32)
    df['is_first'] = df['is_first'].astype(bool)
    df['f_isdigit'] = df['f_isdigit'].astype(bool)
    df['f_isdate'] = df['f_isdate'].astype(bool)
    df['f_iszipcode'] = df['f_iszipcode'].astype(bool)
    df['f_isalpha'] = df['f_isalpha'].astype(bool)
    df['f_isalnum'] = df['f_isalnum'].astype(bool)
    df['f_isnum'] = df['f_isnum'].astype(bool)
    df['f_isnumsym'] = df['f_isnumsym'].astype(bool)
    df['f_isnumreal'] = df['f_isnumreal'].astype(bool)


    return df 

def load_key_classifier(pkl_file="ktp_keys.pkl"):
    clf = None 
    with open(pkl_file, "rb") as of:
        clf = pickle.load(of)
    assert clf
    return clf 

def predict_key_fields(df, clf):
    """ predict f_y """
    X = df[['f_isdigit', 'f_isdate', 'f_iszipcode', 'f_isalpha', 'f_isalnum',
        'f_isnum', 'f_isnumsym', 'f_isnumreal', 'f_nik', 'f_nama', 'f_jenis',
        'f_tempat', 'f_alamat', 'f_rtrw', 'f_keldesa', 'f_kecamatan', 'f_agama',
        'f_status_perkawinan', 'f_pekerjaan', 'f_kewarganegaraan', 'f_berlaku',
        'f_len', 'ncx', 'ncy']]
    df['f_ydash'] = clf.predict(X)

    def clean_y(x):
        #if is_first then we force it to 0
        #if we already have f_y we use that
        #only otherones we use f_ydash
        if x['is_first']:
            return 0

        if x['f_y'] > 0:
            return x['f_y']
        
        return x['f_ydash']

    df['f_y'] = df.apply(clean_y, axis=1)
    return df 

def predict_value_fields(df):
    """ v_y is the value field. if f_y is !=0 return 0. otherwise predict value."""
    df['v_y'] = 0
    ncy_mean = df[df['f_y'] != 0]['nh'].mean()

    from functools import partial

    def assign_value_fields(df, x):
        if x['f_y'] != 0:
            return 0 #key field
            
        file = x['file']
        res = df[(df['file'] == file) & (df['f_y'] != 0) & (df['ncy'] > x['ncy'] - (ncy_mean*2) ) & (df['ncy'] < x['ncy'] + ncy_mean)] #we allow upper margin (2x), but not lower (0.5). 
        
        if len(res) > 0:
            min_tag = res.iloc[0]['f_y']
            min_dist = 2.0
            
            for _, r in res.iterrows():
                d = math.sqrt((x['ncy']-r['ncy'])**2) #only y distance
                if d < min_dist:
                    min_tag = r['f_y']
                    min_dist = d
            return min_tag
            
        return 0

    df['v_y'] = df.apply(partial(assign_value_fields, df), axis=1)
    return df 

def predict_value(df):
    def closest_one(df, pt):
        dx = df.copy()
        
        dx['dist'] = dx.apply(lambda x: math.sqrt( (x['ncx']-pt[0])**2 + (x['ncy']-pt[1])** 2), axis=1)
        return dx['dist'].idxmin()


    def postprocess(dg, x):
        vy = x['v_y']
        file = x['file']
        
        if vy == 0:
            return 0 
        
        single_value = False #vy in [1, 4, 10, 12]

        dk = dg[(dg['file'] == file) & (dg['f_y'] == vy)]    
        if len(dk) == 0:
            return 0 #there is no key field. we can't match this to anything.
        
        key = dk.iloc[0]

        #VALUE
        dx = dg[(dg['f_y'] == 0) & (dg['v_y'] == vy) & (~dg['is_first'])] 

        c = vy 

        if c == 1: #NIK
            dx = dx[(~dx['f_issym']) & (dx['f_len'] > 5) & (dx['f_isdigit'])]  
        elif c == 2: #NAMA
            dx = dx[~dx['f_issym']]
        elif c == 4: #JENIS
            #dx = dx[dx['f_isalpha']]
            dx = dx[~(dx['text'] == 'Gol') & ~(dx['text'] == 'Darah') & ~(dx['text'] == 'Gol.')]
        elif c == 5: #TEMPAT
            dx = dx[dx['f_isalpha'] | dx['f_isdate']]
        elif c == 7: #RTRW
            dx = dx[~(dx['text'] == ':')]
            dx = dx[dx['f_isdigit'] | dx['f_issym']]
        elif c == 10: #AGAMA
            dx = dx[~(dx['text'] == ':')]
            dx = dx[dx['f_isalpha']]
        elif c == 11: #STATUS_PERKAWINAN
            dx = dx[~(dx['text'] == ':')]
            dx = dx[dx['ncx'] < 0.6]
        elif c == 12: #PEKERJAAN
            dx = dx[dx['f_isalpha']]
            dx = dx[dx['ncx'] < 0.6]
        elif c == 13: #KEWARGANEGARAAN
            dx = dx[dx['f_isalpha']] 
        elif c == 14: #BERLAKU
            dx = dx[dx['ncx'] < 0.6]
            dx = dx[dx['f_isdate'] | (dx['text'] == 'SEUMUR') | (dx['text'] == 'HIDUP')]

        if len(dx) > 0:
            if single_value:
                idx = closest_one(dx, (key['ncx'], key['ncy']))
                return c if x.index.equals(idx)  else 0
            else:
                return c if x.name in dx.index else 0
        return 0
                
    df['v_y2'] = df.apply(partial(postprocess, df), axis=1)

    def postprocess_text(dx):
        if dx['v_y2'] == 1:
            return ''.join(filter(str.isdigit, dx['text']))

        return dx['text']

    df['text'] = df.apply(postprocess_text, axis=1)
    return df 


def plot_label(img, label, cx,cy,w,h,color=(0, 181, 20, 255), thickness=2):

    cv2.line(img, (cx-w//2, cy-h//2), (cx+w//2,cy-h//2), color, thickness)
    cv2.line(img, (cx+w//2,cy-h//2),  (cx+w//2,cy+h//2), color, thickness)
    cv2.line(img, (cx+w//2,cy+h//2), (cx-w//2,cy+h//2), color, thickness)
    cv2.line(img, (cx-w//2,cy+h//2), (cx-w//2,cy-h//2), color, thickness)
    
    cv2.putText(img, label, (cx-w//2, cy-h//2), fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=1,color=(255, 0, 0, 255), thickness=1)

def plot_document(img, df):
    classes = ['TEXT',
                'NIK',
                'NAMA',
                'TEXTUNUSED',
                'JENIS_KELAMIN',
                'TEMPAT/TGL_LAHIR',
                'ALAMAT',
                'RT_RW',
                'KEL_DESA',
                'KECAMATAN',
                'AGAMA',
                'STATUS_PERKAWINAN',
                'PEKERJAAN',
                'KEWARGANEGARAAN',
                'BERLAKU_HINGGA']
    
    for ix,rr in df.iterrows():
        if 'v_y2' in rr:
            label = f"{classes[rr['v_y2']] if rr['f_y'] == 0 else classes[rr['f_y']]}"
        else:
            label = f"{classes[rr['f_y']]}"
        label = label if len(rr['text']) > 1 else ''
        cx = int(rr['cx']* img.shape[1])
        cy = int(rr['cy']* img.shape[0])
        w = int(rr['w'] * img.shape[1])
        h = int(rr['h'] * img.shape[0])

        color = (0,200,0,255) if rr['f_y'] == 0 else (0,0,200,255)
        plot_label(img, label, cx, cy, w, h, color)

    return img


def plot_json(img, df):
    
    for ix,rr in df.iterrows():
        label = f"{rr['text']}"
        cx = int(rr['cx']* img.shape[1])
        cy = int(rr['cy']* img.shape[0])
        w = int(rr['w'] * img.shape[1])
        h = int(rr['h'] * img.shape[0])

        color = (0,200,0,255) if len(rr['text']) <= 1 else (0,0,200,255)
        plot_label(img, label, cx, cy, w, h, color)

    return img

def plot_figure(file, df, show=False, waitkey=0, out_file=None):
    #load the image
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #create the canvas to draw boxes
    scale = 1
    #img2 = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale))
    img2 = cv2.resize(img, (1600, 1200))
    img = np.ones((img2.shape), dtype=img.dtype) * 255
    canvas = cv2.addWeighted(img, 0.5, img2, 0.5, 0.0)

    #draw boxes
    img = canvas.copy()
    img = plot_document(img, df)

    img2 = canvas.copy()
    img2 = plot_json(img2, df)

    #show the image
    if show:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("Result",cv2.WINDOW_NORMAL)
        cv2.imshow("Result", img)

        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("Raw Text", cv2.WINDOW_NORMAL)
        cv2.imshow("Raw Text", img2)
        cv2.waitKey(waitkey)
    
    if out_file:
        cv2.imwrite(out_file, img)


def to_json(df):
    classes = ['TEXT',
                'NIK',
                'NAMA',
                'TEXTUNUSED',
                'JENIS_KELAMIN',
                'TEMPAT/TGL_LAHIR',
                'ALAMAT',
                'RT_RW',
                'KEL_DESA',
                'KECAMATAN',
                'AGAMA',
                'STATUS_PERKAWINAN',
                'PEKERJAAN',
                'KEWARGANEGARAAN',
                'BERLAKU_HINGGA']
    res = defaultdict(str)

    for i in range(1,len(classes)):
        if i == 3:
            continue 
        dv = df[df['v_y2'] == i]
        res[classes[i].lower()] = ' '.join(dv['text'].tolist())
    return res 

def main(path, pkl_path):
    df = load_data(path)
    df = process_features(df)
    df = predict_keys(df, load_key_classifier(pkl_path))
    df = predict_value_estimates(df)
    df = predict_value(df)
    plot_figure(str(DATA_DIR/imgfile), df)
    return to_json(df)

    
        
def process_ktp(jsonfile, clfweights="src/ktp_expertsys/ktp_keys.pkl"):
    df = load_data([jsonfile])
    df = process_features(df)
    df = predict_key_fields(df, load_key_classifier(clfweights))
    df = predict_value_fields(df)
    df = predict_value(df)
    return df 

if __name__ == "__main__":
    DATA_DIR = Path("data/interim/2022-01-06_combined_orc_kyc-key-fields/KTP")
    file = "KTP29.json"
    imgfile = "KTP29.jpeg"
    df = process_ktp(DATA_DIR/file)
    print(to_json(df))
    plot_figure(str(DATA_DIR/imgfile), df, show=True)

