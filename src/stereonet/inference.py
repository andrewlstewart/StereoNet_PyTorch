"""
Script to instantiate a StereoNet model + run inference over the SceneFlow dataset test set.

Mostly used to visualize batches.
"""

from typing import List
from pathlib import Path
import argparse
import statistics

import torch
import torch.nn.functional as F

from stereonet.model import StereoNet
import stereonet.utils as utils
from stereonet import stereonet_types as st


def parse_test_args() -> argparse.Namespace:
    """
    Parser for arguments related to testing.
    """
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--checkpoint_path', type=Path, help="Model checkpoint path to load.")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--gpu', action='store_true', help="Flag to use gpu for inference.")

    # Dataset parameters
    parser.add_argument('--sceneflow_root', type=Path, help="Root path containing the sceneflow folders containing the images and disparities.")

    # Inference parameters
    parser.add_argument('--plot_figure', action='store_true', help="Flag to plot ground truth and predictions.")

    return parser.parse_args()


def get_disp_above_val(parent_path: Path, max_disp: float) -> float:
    """
    Helper function to find the average number of pixels greater than the provided maximum dispisparity.
    """
    mean_above_max_disp: List[float] = []
    # maxes = []
    for disp_path in parent_path.rglob('*.pfm'):
        disp, _ = utils.pfm_loader(disp_path)
        # maxes.append(disp.max())
        mean_above_max_disp.append((disp > max_disp).mean().item())

    return statistics.mean(mean_above_max_disp)


def sceneflow_inference() -> None:  # pylint: disable=missing-function-docstring
    """
    Compute the EPE
    """

    args = parse_test_args()

    device = torch.device("cuda:0" if args.gpu else "cpu")

    model = StereoNet.load_from_checkpoint(args.checkpoint_path)
    model.to(device)
    model.eval()

    val_transforms: List[st.TorchTransformer] = [utils.Rescale()]
    val_dataset = utils.SceneflowDataset(args.sceneflow_root, string_include='TEST', transforms=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    max_disp = model.candidate_disparities if model.mask else float('inf')

    loss = 0.0
    loss_maxdisp = 0.0
    for batch_idx, batch in enumerate(val_loader):
        for name, tensor in batch.items():
            batch[name] = tensor.to(device)
        with torch.no_grad():
            output = model({'left': batch['left'], 'right': batch['right']})
        loss += F.l1_loss(batch['disp_left'], output).item()
        loss_maxdisp += F.l1_loss(batch['disp_left'][batch['disp_left'] < max_disp], output[batch['disp_left'] < max_disp]).item()

        if args.plot_figure:
            utils.plot_figure(batch['left'][0], batch['right'][0], batch['disp_left'][0], output[0])

    print(f'Validation EPE: {loss/batch_idx}')
    print(f'Validation EPE: {loss_maxdisp/batch_idx}')


if __name__ == "__main__":
    sceneflow_inference()