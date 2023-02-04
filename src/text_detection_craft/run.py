import os
import cv2
import torch
import argparse
from torch.autograd import Variable
from net import *
from utils import *
from config import config as conf


OUTPUT_FOLDER = 'craft'

def detect_text(model, image, canvas_size, text_threshold, link_threshold, low_text, mag_ratio, device):
    img_resized, target_ratio, _ = resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  
    x = x.to(device)
    with torch.no_grad():
        y, features = model(x)
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    boxes, labels, mapper = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    return boxes, score_text

def load(save_path=None, device='cpu'):
    model = get_craft()
    if device == 'cpu':
        model.load_state_dict(copyStateDict(torch.load(save_path, map_location='cpu')))
    else:
        model.load_state_dict(copyStateDict(torch.load(save_path)))
    return model.to(device)

def get_images(dir=None):
    image_items = [os.path.join(dir, i) for i in os.listdir(dir)] 
    return image_items, len(image_items)

def main(args=None):
    device = 'cpu' if not args.gpu else conf.DEVICE
    mk_dir(args.output_directory, OUTPUT_FOLDER)
    model = load(args.trained_model, device)
    model.eval()
    image_items, image_items_length = get_images(args.input_directory)
    for ind, image_path in enumerate(image_items):
        print(f"Image {ind+1}/{image_items_length}: {image_path}", end='\n')
        image = loadImage(image_path)
        bboxes, _ = detect_text(model, image, args.canvas_size, args.text_threshold, args.link_threshold, args.low_text, args.mag_ratio, device)
        saveResult(image_path, image[:,:,::-1], bboxes, dirname=args.output_directory + f'/{OUTPUT_FOLDER}')
        if args.demo:
            break

if __name__ == '__main__':
    # Fetching arguments.
    parser = argparse.ArgumentParser(description='Text Detection')
    parser.add_argument('-m', '--trained_model', type=str, default='../../runs/downloaded_models/craft_mlt_25k.pth', metavar="\b", help='Path to trained model')
    parser.add_argument('-i', '--input_directory', type=str, default='../../data/interim/2021-06-14_ocr_kyc-pdfs', metavar="\b", help='Path to directory containing images')
    parser.add_argument('-o', '--output_directory', type=str, default='../../data/output', metavar="\b", help='Path to the directory in which output will be saved')
    parser.add_argument('-g', '--gpu', type=str2bool, default='n', metavar="\b", help='Use GPU?')
    parser.add_argument('-text', '--text_threshold', default=0.7, type=float, metavar="\b", help='Text confidence threshold')
    parser.add_argument('-low', '--low_text', default=0.4, type=float, metavar="\b", help='Text low-bound score')
    parser.add_argument('-lnk', '--link_threshold', default=0.4, type=float, metavar="\b", help='Link confidence threshold')
    parser.add_argument('-c', '--canvas_size', default=1280, type=int, metavar="\b", help='Image size for inference')
    parser.add_argument('-mr', '--mag_ratio', default=1.5, type=float, metavar="\b", help='Image magnification ratio')
    parser.add_argument('-d', '--demo', default='y', type=str2bool, metavar="\b", help='Try out with one image only')
    args = parser.parse_args()
    main(args)
