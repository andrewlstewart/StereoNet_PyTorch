"""
"""

from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.model import StereoNet
import src.utils as utils


class Tuplelify(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = values.split(",")
        if float(values[0]) == int(values[0]):
            values = tuple(int(v) for v in values)
        else:
            values = tuple(float(v) for v in values)
        setattr(namespace, self.dest, values)


def parse_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sceneflow_root', type=Path, help="Root path containing the sceneflow folders containing the images and disparities.")
    parser.add_argument('--checkpoint_path', type=Path, help="Model checkpoint path to load.")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--gpu', action='store_true', help="Flag to use gpu for inference.")

    return parser.parse_args()


def main():

    args = parse_test_args()

    device = torch.device("cuda:0" if args.gpu else "cpu")

    model = StereoNet(k_downsampling_layers=3, k_refinement_layers=3, candidate_disparities=192).load_from_checkpoint(args.checkpoint_path)
    model.to(device)
    model.eval()

    val_transforms = [utils.ToTensor(), utils.Rescale()]
    val_dataset = utils.SceneflowDataset(args.sceneflow_root, string_include='TEST', transforms=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    samples = 0
    loss = 0
    for batch in val_loader:
        for name, tensor in batch.items():
            batch[name] = tensor.to(device)
        with torch.no_grad():
            output = model((batch['left'], batch['right']))
        loss += F.l1_loss(batch['disp'], output)
        samples += batch['disp'].size()[0]

    print(f'Validation EPE: {loss/samples}')

    print('stall')


if __name__ == "__main__":
    main()
