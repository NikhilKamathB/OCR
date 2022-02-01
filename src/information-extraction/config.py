import torch
from easydict import EasyDict 

config = EasyDict()

# data.
config.DATA_DIR_RAW = '../../../data/raw'
config.DATA_DIR_PROCESSED = '../../../data/processed/information-extraction'
config.DATA_TRAIN_OCR_ENGINE = [
    f"{config.DATA_DIR_RAW}/train-1"
]
config.DATA_TRAIN_IE = [
    f"{config.DATA_DIR_RAW}/train-2"
]
config.DATA_TEST = [
    f"{config.DATA_DIR_RAW}/test-1"
]
config.DATA_TEST_STANDALONE = [
    f"{config.DATA_DIR_RAW}/test-2"
]
config.LABEL = {'ADDRESS': 1, 'COMPANY': 2, 'DATE':3, 'TOTAL': 4, 'OTHERS': 0}
config.IND_LABEL = {str(v): k for k, v in config.LABEL.items()}
config.TEST_SPLIT = 0.1
config.VAL_SPLIT = 0.1

# data cleaning and pre-processing configurations.
config.RESIZE_WIDTH = 720
config.RESIZE_HEIGHT = 1080

# model hyperparameters.
config.LEARNING_RATE = 1e-3
config.EPOCHS = 750
config.BATCH_SIZE = 128

# logging.
config.LOG_PATH = '../../../logs'
config.VERBOSE = True

# model logging.
config.MODEL_DIR = '../../../runs'
config.MODEL_SAVED_PATH = '../../../runs/models'

# device.
config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")