import torch
from easydict import EasyDict 

config = EasyDict()

# model to train on.
config.__model_name__ = 'CRAFT'

# plot related entities.
config.DISPLAY_COLUMNS = 4
config.DISPLAY_ROWS = 5

# data augmentation related entities.
config.TRANSFORM_SIZE = 192

# model hyperparameters.
config.LEARNING_RATE = 1e-3
config.MOMENTUM = 0.9
config.EPOCHS = 1
config.TEST_BATCH_SIZE = 32
config.VALIDATION_BATCH_SIZE = 32
config.TRAIN_BATCH_SIZE = 32
config.PATIENCE = 10

# logging.
config.VERBOSE = True
config.VERBOSE_STEP = 500
config.TEST_RUN = 1

# path to data attributes - stanford cars.
config.DATA_DIR_RAW = '../../data/raw'
config.DATA_DIR_INTERIM = '../../data/interim'
config.DATA_DIR_PROCESSED = '../../data/processed'

# wandb specifics.
config.WANDB_BASE_DIR = '../../runs/'

# path to model attributes.
config.MODEL_DIR = '../../runs/'
config.MODEL_SAVED_PATH = '../../runs/saved_models/'

# pick device.
config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")