import argparse
from utils import str2bool
from inference import Inference


def main(args):
    _ = Inference(
        saved_model=args.trained_model_path,
        input_image_directory=args.input_image_directory,
        number_of_images_to_infer=args.number_of_images_to_infer,
        shuffle=args.shuffle,
        wrtie_output=args.write_output,
        output_directory=args.output_directory,
        model_name=args.model_name,
        device=args.device,
        processor_pretrained_path=args.trocr_processor_predifined_path,
        verbose=args.verbose
    ).infer()

if __name__ == '__main__':
    # Fetch arguments from command line
    parser = argparse.ArgumentParser(description='Text recognition module.')
    parser.add_argument("-m", "--model_name", type=str, default="TrOCRModel", metavar='\b', help="Name of the model.")
    parser.add_argument("-p", "--trained_model_path", type=str, default="microsoft/trocr-base-handwritten", metavar='\b', help="Path to trained model.")
    parser.add_argument("-pp", "--trocr_processor_predifined_path", default="microsoft/trocr-base-handwritten", type=str, metavar='\b', help="TrOCRProcessor predefined path.")
    parser.add_argument("-d", "--device", type=str, default="cpu", metavar='\b', help="Device to use | 'cpu', 'cuda', 'mps'.")
    parser.add_argument("-i", "--input_image_directory", type=str, metavar='\b', help="Path to input image directory.")
    parser.add_argument("-w", "--write_output", type=str2bool, default='n', metavar="\b", help="Write output to disk.")
    parser.add_argument("-o", "--output_directory", type=str, metavar='\b', help="Path to output directory that holds results.")
    parser.add_argument("-n", "--number_of_images_to_infer", type=int, default=1, metavar='\b', help="Total number of images to process.")
    parser.add_argument('-s', '--shuffle', type=str2bool, default='n', metavar="\b", help="Shuffle images in the aforementioned directory.")
    parser.add_argument('-v', '--verbose', type=str2bool, default='n', metavar="\b", help="Verbose.")
    main(args=parser.parse_args())