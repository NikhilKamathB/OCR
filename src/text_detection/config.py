from easydict import EasyDict 


config = EasyDict()

# data.
config.DATA_DIR_RAW = '../../data/raw'
config.DATA_DIR_RAW_TRAIN = f'{config.DATA_DIR_RAW}/train'
config.DATA_DIR_RAW_TEST = f'{config.DATA_DIR_RAW}/test'
config.DATA_DIR_ANNOTATION_TRAIN = f'{config.DATA_DIR_RAW}/train'
config.DATA_DIR_ANNOTATION_TEST = f'{config.DATA_DIR_RAW}/test'