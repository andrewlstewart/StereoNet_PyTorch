"""
Several type aliases to make type hints cleaner.
"""

from typing import Union, Dict, List

from abc import ABC, abstractmethod

import torch
import numpy as np
import numpy.typing as npt


Number = Union[int, float]

Sample_Numpy = Union[Dict[str, npt.NDArray[np.uint8]], Dict[str, npt.NDArray[np.float32]]]
Sample_Torch = Dict[str, torch.Tensor]
Sample = Union[Sample_Numpy, Sample_Torch]


class TorchTransformer(ABC):
    @abstractmethod
    def __call__(self, x: Sample_Torch) -> Sample_Torch:
        pass


class NumpyToTorchTransformer(ABC):
    @abstractmethod
    def __call__(self, x: Sample_Numpy) -> Sample_Torch:
        pass


TorchTransformers = Union[TorchTransformer, List[TorchTransformer]]
