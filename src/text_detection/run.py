import argparse


def main(args=None):
    pass


if __name__ == '__main__':
    # Fetch arguments from command line
    parser = argparse.ArgumentParser(description='Text detection module.')
    parser.add_argument("-m", "--trained-model", type=str, metavar='\b', help="Path to trained model.")
    parser.add_argument("-i", "--input-image-directory", type=str, metavar='\b', help="Path to input image directory.")