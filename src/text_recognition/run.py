import os
import cv2
import torch
import argparse
from torch.autograd import Variable
from net import *
from utils import *
from config import config as conf


OUTPUT_FOLDER = 'crnn'

def load(save_path=None, device='cpu'):
    model = get_crnn()
    print(torch.load(save_path, map_location=device).keys())
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model.to(device)

def main(args=None):
    device = 'cpu' if not args.gpu else conf.DEVICE
    mk_dir(args.output_directory, OUTPUT_FOLDER)
    model = load(args.trained_model, device)
    pass

if __name__ == '__main__':
    # Fetching arguments.
    parser = argparse.ArgumentParser(description='Text Detection')
    parser.add_argument('-m', '--trained_model', type=str, default='../../runs/downloaded_models/VGG-LSTM.pth', metavar="\b", help='Path to trained model directory')
    parser.add_argument('-i', '--input_directory', type=str, default='../../data/raw/text_recognition_images', metavar="\b", help='Path to directory containing images')
    parser.add_argument('-o', '--output_directory', type=str, default='../../data/output', metavar="\b", help='Path to the directory in which output will be saved')
    parser.add_argument('-v', '--vocabulary', type=str, default=conf.VOCAB, metavar="\b", help='Vocabulary as string')
    parser.add_argument('-g', '--gpu', type=str2bool, default='n', metavar="\b", help='Use GPU?')
    parser.add_argument('-d', '--demo', default='y', type=str2bool, metavar="\b", help='Try out with one image only')
    args = parser.parse_args()
    main(args)
