import torch
from easydict import EasyDict 

config = EasyDict()

# model to train on.
config.__model_name__ = 'CRNN'
config.MODEL_NAME = 'CUSTOM_VGG_CRNN'

# plot related entities.
config.DISPLAY_COLUMNS = 4
config.DISPLAY_ROWS = 5

# data augmentation related entities.
config.TRANSFORM_SIZE_HEIGHT = 50
config.TRANSFORM_SIZE_WIDTH = 200
config.TEXT_MAX_LENGTH = 25

# model hyperparameters.
config.LEARNING_RATE = 1e-3
config.MOMENTUM = 0.9
config.WEIGHT_DECAY = 1e-5
config.CLIP_NORM = 5
config.EPOCHS = 100
config.TEST_BATCH_SIZE = 32
config.VALIDATION_BATCH_SIZE = 32
config.TRAIN_BATCH_SIZE = 32
config.OVERFIT_BATCH_SIZE = 32
config.PATIENCE = 15

# logging.
config.VERBOSE = True
config.VERBOSE_STEP = 00
config.TEST_RUN = 1

# path to data attributes - ocr - text recognition.
config.DATA_DIR_RAW = '../data/raw'
config.DATA_DIR_INTERIM = '../data/interim'
config.DATA_DIR_PROCESSED = '../data/processed'
config.DATA_SPLIT_DIR = '../data/processed/text_recognition'
config.INTERIM_2021_08_10_text_recognition_csv = '../data/interim/2021-08-10_text_recognition/text_recognition_annots.csv'
config.INTERIM_2021_08_17_text_recognition_mjsynth_csv = '../data/interim/2021-08-17_text_recognition_mjsynth_images/text_recognition_mjsynth_annots.csv'
config.ANNOTATIONS = [config.INTERIM_2021_08_10_text_recognition_csv, config.INTERIM_2021_08_17_text_recognition_mjsynth_csv]

# wandb specifics.
config.WANDB_BASE_DIR = '../runs/'

# path to model attributes.
config.MODEL_DIR = '../runs/'
config.MODEL_DOWNLOADED_PATH = '../runs/downloaded_models/'
config.MODEL_SAVED_PATH = '../runs/saved_models/'
config.MODEL_TEST_PATH = '../runs/saved_models/RESNET18_CRNNZQOS8Z0OP3.pth'

# pick device.
config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")