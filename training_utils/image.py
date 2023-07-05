import numpy as np
from typing import Tuple
from PIL import Image


def resize_image(img: np.ndarray, img_shape: Tuple):
    """ Resize the image to the given shape

    Args:
        img: Image as np.ndarray
        img_shape: Desired shape of the image (2-d shape)

    Returns:

    """
    img = (np.asarray(img.resize(img_shape, Image.ANTIALIAS))/255).astype(np.float32)
    img = np.moveaxis(img, -1, 0) # move the last dimension to the first, since pytorch is "channels first"

    return img