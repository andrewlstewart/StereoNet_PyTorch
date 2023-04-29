"""
"""

from typing import Tuple, Union
from pathlib import Path
import re
import os

import numpy as np
import numpy.typing as npt
from skimage import io
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402


def pfm_loader(path: Union[Path, str]) -> Tuple[npt.NDArray[np.float32], float]:
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
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


def exr_loader(path: Union[Path, str]) -> npt.NDArray[np.float32]:
    """
    Load an image from a path using opencv and return a np.float32 numpy array.
    """
    img: npt.NDArray[np.float32] = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img
