import io
import os
from PIL import Image


def mk_dir(dir=None):
    if not os.path.isdir(dir):
        os.mkdir(dir)

def image_to_byte_array(image: Image, format: str = None):
    imgByteArr = io.BytesIO()
    if not format:
        image.save(imgByteArr, format=image.format)
    else:
        image.save(imgByteArr, format=format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr