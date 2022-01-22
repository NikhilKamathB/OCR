import torch
from easydict import EasyDict 

config = EasyDict()

# data.
config.DATA_DIR_RAW = '../data/raw/information-extraction'
config.DATA_DIR_PROCESSED = '../data/processed/information-extraction'

# data cleaning and pre-processing configurations.
config.RESIZE_WIDTH = 256
config.RESIZE_HEIGHT = 256

# model hyperparameters.
config.LEARNING_RATE = 1e-3
config.EPOCHS = 100
config.TEST_BATCH_SIZE = 32
config.VALIDATION_BATCH_SIZE = 32
config.TRAIN_BATCH_SIZE = 32
config.OVERFIT_BATCH_SIZE = 32

# logging.
config.LOG_PATH = '../logs'
config.VERBOSE = True

# model logging.
config.MODEL_DIR = '../runs'
config.MODEL_SAVED_PATH = '../runs/models'

# device.
config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")