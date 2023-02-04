import os
import cv2
import wandb
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from .config import config as conf


class CTCLabelConverter:

    def __init__(self, characters):
        # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
        self.dict_char_to_index = {i: ind+1 for ind, i in enumerate(characters)}
        self.dict_char_to_index[' '] = 0
        self.dict_index_to_char = {self.dict_char_to_index[i]: i for i in self.dict_char_to_index.keys()}
        self.characters = [' '] + characters  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, max_length=25):
        # The index used for padding (=0) would not affect the CTC loss calculation.
        text_encoded = [self.dict_char_to_index[char] for char in text]
        if len(text_encoded) <= max_length:
            return text_encoded + [0] * (max_length - len(text_encoded)), [len(text)]
        else:
            return text_encoded[:min(len(text_encoded), max_length)], [max_length]

    def decode(self, text_index_list):
        return ''.join(self.dict_index_to_char[i] for i in text_index_list)

def generate_vocab(words=None):
    vocab_char = set()
    for word in tqdm(words):
        for i in list(word):
            vocab_char.add(i)
    return list(vocab_char)

def str2bool(v):
    return v.lower() in ("yes", "Yes", "YES", "y", "true", "True", "TRUE", "t", "1")

def mk_dir(path=None, output_folder=None):
    if not os.path.isdir(path):
        os.mkdir(path)
        os.mkdir(path+f'/{output_folder}')
    if not os.path.isdir(path+f'/{output_folder}'):
        os.mkdir(path+f'/{output_folder}')

def create_df(path_list=None):
    assert isinstance(path_list, list), 'path_list must be a list of paths to the csv file'
    df_list = []
    for path in path_list:
        df_list.append(pd.read_csv(path))
    df = pd.concat(df_list).reset_index(drop=True)
    return df

def get_df(path=None, df=None):
    assert path is not None or df is not None, 'Mention either path or df, both cannot be None'
    if path:
        path = conf.INTERIM_2021_08_10_text_recognition_csv if path == 'default' else path
        df = pd.read_csv(path)
    df.dropna(axis=0, subset=['label'], inplace=True)
    df['image_path_nbs'] = df['image_path'].apply(lambda x: x[3:])
    df['parent_image_path_nbs'] = df['parent_image_path'].apply(lambda x: x[3:] if isinstance(x, str) else 'N/A')
    df['parent_json_path_nbs'] = df['parent_json_path'].apply(lambda x: x[3:] if isinstance(x, str) else 'N/A')
    return df

def get_split(df=None, train_split=0.6, val_split=0.2):
    train_df = df[: int(len(df)*train_split)]
    val_df = df[int(len(df)*train_split): int(len(df)*(train_split+val_split))]
    test_df = df[int(len(df)*(train_split+val_split)): ]
    return train_df, val_df, test_df

def split_data(df, train_split=0.6, val_split=0.2):
    df = df.sample(frac=1, random_state=101)
    if os.path.exists(os.path.join(conf.DATA_SPLIT_DIR, 'train.csv')) and os.path.exists(os.path.join(conf.DATA_SPLIT_DIR, 'val.csv')) and os.path.exists(os.path.join(conf.DATA_SPLIT_DIR, 'test.csv')):
        train_df, val_df, test_df = pd.read_csv(os.path.join(conf.DATA_SPLIT_DIR, 'train.csv')), pd.read_csv(os.path.join(conf.DATA_SPLIT_DIR, 'val.csv')), pd.read_csv(os.path.join(conf.DATA_SPLIT_DIR, 'test.csv'))
    else:
        train_df, val_df, test_df = get_split(df=df, train_split=train_split, val_split=val_split)
        train_df.to_csv(os.path.join(conf.DATA_SPLIT_DIR, 'train.csv'))
        val_df.to_csv(os.path.join(conf.DATA_SPLIT_DIR, 'val.csv'))
        test_df.to_csv(os.path.join(conf.DATA_SPLIT_DIR, 'test.csv'))
    return train_df, val_df, test_df

def display_images(image_set=None, rows=5, columns=4, figsize=(7, 7), need_capture=True, label_convertor=None, wandb_title='Test output', wandb_needed=False, is_ploting_prediction=False):
    assert image_set is not None, "'image_set' arg cannot be None."
    fig = plt.figure(figsize=figsize)
    for i in range(1, rows * columns + 1):
        image = None
        if need_capture:
            image = cv2.imread(image_set[i-1][0])[:, :, ::-1]
            title = f"{image_set[i-1][1]}"
        else:
            image = image_set[i-1][0]
            np.clip(image, 0, 1, out=image)
            if is_ploting_prediction:
                title = f"Predicted -> {label_convertor.decode(image_set[i-1][1])}\nTrue -> {label_convertor.decode(image_set[i-1][2])}"
            else:
                title = f"{label_convertor.decode(image_set[i-1][1])} -> Length={len(image_set[i-1][1])}"
        fig.add_subplot(rows, columns, i)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
    fig.tight_layout(pad=2.0)
    if wandb_needed:
        wandb.log({wandb_title: fig})
    plt.show()
    return None

def visualize(data=None, rows=5, columns=4, figsize=(17, 17), labels=None, true_labels=None,  label_convertor=None, wandb_needed=False, is_ploting_prediction=False):
    assert data is not None, "'data' arg cannot be None."
    image_set = []
    need_capture = True
    if isinstance(data, pd.DataFrame):
        df = data.sample(n=rows*columns)
        for _, row in df.iterrows():
            image_set.append((row["image_path_nbs"], row["label"]))
    elif isinstance(data, torch.Tensor):
        for i in range(data.size(0)):
            image = data[i].numpy()
            image = np.transpose(image, (1, 2, 0))
            label = labels[i].tolist()
            if is_ploting_prediction:
                true_label = true_labels[i].tolist()
                image_set.append((image, label, true_label))
            else:
                image_set.append((image, label))
            if i+1 == rows*columns:
                break
        need_capture = False
    display_images(image_set=image_set, rows=rows, columns=columns, figsize=figsize, need_capture=need_capture, label_convertor=label_convertor, wandb_title='Test output', wandb_needed=wandb_needed, is_ploting_prediction=is_ploting_prediction)

def get_loss(loss='ctc'):
    if loss == 'ctc':
        return torch.nn.CTCLoss(blank=0, zero_infinity=True)

def get_misc(model=None, lr=None, momentum=None, weight_decay=None, patince=10, loss='ctc'):
    assert model is not None, "'model' arg cannot be None."
    lr = conf.LEARNING_RATE if lr is None else lr
    momentum = conf.MOMENTUM if momentum is None else momentum
    weight_decay = conf.WEIGHT_DECAY if weight_decay is None else weight_decay
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, weight_decay=weight_decay, momentum=momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = get_loss(loss=loss)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patince)
    return optimizer, criterion, scheduler