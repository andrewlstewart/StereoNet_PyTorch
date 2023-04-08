"""
"""

from typing import Tuple, Union
from pathlib import Path
import re

import numpy as np
import numpy.typing as npt
from skimage import io


def PFM_loader(path: Union[Path, str]) -> Tuple[npt.NDArray[np.float32], float]:
    """
    This function was entirely written by the the Freiburg group
    https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py

    Read in a PFM formated file and return a image/disparity
    """
    with open(path, 'rb') as file:
        header = file.readline().rstrip()
        if header.decode("ascii") == 'PF':
            color = True
        elif header.decode("ascii") == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data: npt.NDArray[np.float32] = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def image_loader(path: Union[Path, str]) -> npt.NDArray[np.uint8]:
    """
    Load an image from a path using skimage.io and return a np.uint8 numpy array.
    """
    img: npt.NDArray[np.uint8] = io.imread(path)
    return img
