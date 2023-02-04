from .models.crnn import *
from .config import config as conf


def get_model(conf=conf):
    return eval(conf.__model_name__)