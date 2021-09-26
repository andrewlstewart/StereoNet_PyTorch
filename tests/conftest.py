from typing import List, Tuple

import pytest

from stereonet.model import StereoNet


@pytest.fixture(scope="session")
def models() -> List[Tuple[StereoNet, int]]:
    """
    Instantiates a bunch of models so we can test their output sizes.
    """
    models = [
        (StereoNet(k_downsampling_layers=3, k_refinement_layers=1), 398978),  # Tuple of (model, number of trainable parameters)
        (StereoNet(k_downsampling_layers=4, k_refinement_layers=1), 424610),
        (StereoNet(k_downsampling_layers=3, k_refinement_layers=3), 624644),
        (StereoNet(k_downsampling_layers=4, k_refinement_layers=3), 650276)
    ]

    for (model, _) in models:
        model.eval()

    return models
