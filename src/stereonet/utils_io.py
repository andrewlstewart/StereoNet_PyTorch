"""

This script was entirely written by the the Freiburg group
https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py

"""

from typing import Tuple
from pathlib import Path
import re

import numpy as np
import numpy.typing as npt


def readPFM(path: Path) -> Tuple[npt.NDArray[np.float32], float]:
    """
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
