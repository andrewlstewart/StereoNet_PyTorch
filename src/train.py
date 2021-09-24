"""
Script to instantiate a StereoNet model + train on the SceneFlow dataset.
"""

from pathlib import Path
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from src.model import StereoNet
import src.utils as utils


def parse_train_args() -> argparse.Namespace:
    """
    Parser for arguments related to training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--sceneflow_root', type=Path, help="Root path containing the sceneflow folders containing the images and disparities.")
    parser.add_argument('--k_downsampling_layers', type=int, default=3)
    parser.add_argument('--k_refinement_layers', type=int, default=3)
    parser.add_argument('--candidate_disparities', type=int, default=256)

    parser.add_argument('--train_batch_size', default=1, type=int)
    parser.add_argument('--val_batch_size', default=1, type=int)
    parser.add_argument('--min_epochs', type=int, default=10, help="Minimum number of epochs to train.")
    parser.add_argument('--max_epochs', type=int, default=50, help="Maximum number of epochs to train.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed used in the image transforms (not related to image selection in batches)")
    parser.add_argument('--crop_percentage', type=float, default=0.95, help="Factor to rescale all images/disparities.  Required because I don't have enough VRAM.")
    # parser.add_argument('--rescale_size', type=int, default=1, help="Factor to rescale (shrink) the images, 2 corresponds with a w/2 and h/2 or 4x shrink in image dimensions.")

    parser.add_argument('--num_gpus', type=int, default=0, help="Number of GPUs to use.")

    return parser.parse_args()


def main():  # pylint: disable=missing-function-docstring

    args = parse_train_args()

    # Instantiate model with built in optimizer
    model = StereoNet(k_downsampling_layers=args.k_downsampling_layers, k_refinement_layers=args.k_refinement_layers, candidate_disparities=args.candidate_disparities)

    # Get datasets
    # random_generator = np.random.default_rng(seed=args.random_seed)
    train_transforms = [
        utils.ToTensor(),
        utils.Rescale(),
        utils.CenterCrop(scale=args.crop_percentage)
        # utils.RandomCrop(scale=args.crop_percentage, randomizer=random_generator)
        # utils.RandomResizedCrop(output_size=(int(540/args.rescale_size), int(960/args.rescale_size)),
                                # scale=(0.8/args.rescale_size, 1.2/args.rescale_size),
                                # randomizer=random_generator),
        # TODO: Color jitter?
    ]
    train_dataset = utils.SceneflowDataset(args.sceneflow_root, string_exclude='TEST', transforms=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8, drop_last=False)

    val_transforms = [utils.ToTensor(), utils.Rescale()]
    val_dataset = utils.SceneflowDataset(args.sceneflow_root, string_include='TEST', transforms=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=8, drop_last=False)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger(save_dir=Path.cwd(), name="lightning_logs")
    trainer = pl.Trainer(gpus=args.num_gpus, min_epochs=args.min_epochs, max_epochs=args.max_epochs, logger=logger, callbacks=[lr_monitor], checkpoint_callback=True)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
