"""
Test suite for instantiating StereoNet models and performing simple forward passes.
"""

import unittest

import torch

import src.model as model_zoo


class TestModel(unittest.TestCase):  # pylint: disable=missing-class-docstring

    def setUp(self):
        """
        Instantiates a bunch of models so we can test their output sizes.
        """
        self.models = [
            (model_zoo.StereoNet(k_downsampling_layers=3, k_refinement_layers=1), 398978),  # Tuple of (model, number of trainable parameters)
            (model_zoo.StereoNet(k_downsampling_layers=4, k_refinement_layers=1), 424610),
            (model_zoo.StereoNet(k_downsampling_layers=3, k_refinement_layers=3), 624644),
            (model_zoo.StereoNet(k_downsampling_layers=4, k_refinement_layers=3), 650276)
        ]

        for (model, _) in self.models:
            model.eval()

    def test_forward_sizes(self):
        """
        Test to see if each of the networks produces the correct shape.
        """
        input_data = (torch.rand((2, 3, 540, 960)), torch.rand((2, 3, 540, 960)))  # Tuple of left/right images in batch, channel, height, width

        with torch.no_grad():
            for model, n_params in self.models:
                self.assertEqual(model(input_data).size(), (2, 1, 540, 960))
                self.assertEqual(self.count_parameters(model), n_params)

    @staticmethod
    def count_parameters(model):
        """
        Counts the number of trainable parameters in a torch model
        https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
