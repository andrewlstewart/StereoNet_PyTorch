"""
Several type aliases to make type hints cleaner.
"""

from typing import Union, Dict, List

from abc import ABC, abstractmethod

import torch
import numpy as np
import numpy.typing as npt


Number = Union[int, float]

Sample_Torch = Dict[str, torch.Tensor]
Sample_Numpy = Union[Dict[str, npt.NDArray[np.uint8]], Dict[str, npt.NDArray[np.float32]]]
Sample_General = Union[Sample_Torch, Sample_Numpy]


class NumpyTransformer(ABC):
    @abstractmethod
    def __call__(self, x: Sample_Numpy) -> Sample_Numpy:  # pylint: disable=invalid-name
        pass


class TorchTransformer(ABC):
    @abstractmethod
    def __call__(self, x: Sample_Torch) -> Sample_Torch:  # pylint: disable=invalid-name
        pass


class NumpyToTorchTransformer(ABC):
    @abstractmethod
    def __call__(self, x: Sample_Numpy) -> Sample_Torch:  # pylint: disable=invalid-name
        pass


Transformer = Union[NumpyTransformer, TorchTransformer, NumpyToTorchTransformer]
# Transformer = Union[TorchTransformer, NumpyToTorchTransformer]
Transformers = Union[Transformer, List[Transformer]]
