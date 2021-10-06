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
import matplotlib.pyplot as plt
import numpy as np

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


def stereocam_inference() -> None:  # pylint: disable=missing-function-docstring
    """
    Forward inference using an image captured on the StereoCam
    """

    checkpoint_path = Path(r"C:\Users\andre\Documents\Python\StereoNet_PyTorch_Typed\saved_models\version_42\checkpoints\epoch=27-step=992711.ckpt")
    image_path = Path(r"C:\Users\andre\Documents\Python\StereoNet_PyTorch_Typed\data\image.jpg")

    device = torch.device("cuda:0" if False else "cpu")

    model = StereoNet.load_from_checkpoint(str(checkpoint_path))
    model.to(device)
    model.eval()

    img = utils.image_loader(image_path)

    left_img = img[:, :2028, :]
    right_img = img[:, 2028:, :]

    # left_img = img[:, 2028:, :]
    # right_img = img[:, :2028, :]

    numpy_batch = {'left': left_img, 'right': right_img}
    batch = utils.ToTensor()(numpy_batch)
    tensor_transformers = [utils.Resize((640, 960)), utils.Rescale(), utils.PadSampleToBatch()]
    for transformer in tensor_transformers:
        batch = transformer(batch)

    with torch.no_grad():
        prediction = model(batch)[0].cpu().numpy()
    prediction = np.moveaxis(prediction, 0, 2)

    plt.imshow(prediction, vmin=prediction.min(), vmax=prediction.max())
    plt.show()

    print('stall')


if __name__ == "__main__":
    # sceneflow_inference()
    stereocam_inference()
