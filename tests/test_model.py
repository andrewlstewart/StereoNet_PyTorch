"""
Test suite for instantiating StereoNet models and performing simple forward passes.
"""

from typing import List, Tuple

import torch

from stereonet.model import StereoNet


def test_model_trainable_parameters(models: List[Tuple[StereoNet, int]]):
    """
    Test to see if the number of trainable parameters matches the expected number.
    """
    for model, n_params in models:
        assert (count_parameters(model) == n_params)


def test_forward_sizes(models: List[Tuple[StereoNet, int]]):
    """
    Test to see if each of the networks produces the correct shape.
    """
    input_data = {'left': torch.rand((2, 3, 540, 960)), 'right': torch.rand((2, 3, 540, 960))}

    with torch.no_grad():
        for model, _ in models:
            assert (model(input_data).size() == (2, 1, 540, 960))


def count_parameters(model: StereoNet) -> int:
    """
    Counts the number of trainable parameters in a torch model
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
