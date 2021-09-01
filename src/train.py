"""
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


class Tuplelify(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = values.split(",")
        if float(values[0]) == int(values[0]):
            values = tuple(int(v) for v in values)
        else:
            values = tuple(float(v) for v in values)
        setattr(namespace, self.dest, values)


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sceneflow_root', type=Path, help="Root path containing the sceneflow folders containing the images and disparities.")
    parser.add_argument('--disparity_range', action=Tuplelify, default=(0,192))

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--min_epochs', type=int, default=10, help="Minimum number of epochs to train.")
    parser.add_argument('--max_epochs', type=int, default=50, help="Maximum number of epochs to train.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed used in the image transforms (not related to image selection in batches)")
    parser.add_argument('--crop_size', action=Tuplelify, default=(540,960), help="Size to crop the images to during the random crops for training.")

    parser.add_argument('--num_gpus', type=int, default=0, help="Number of GPUs to use.")

    return parser.parse_args()


def main():

    args = parse_train_args()

    # Instantiate model with built in optimizer
    model = StereoNet(disparity_range=args.disparity_range)

    # Get datasets
    random_generator = np.random.default_rng(seed=args.random_seed)
    train_transforms = [
        utils.ToTensor(),
        utils.Rescale(),
        utils.RandomResizedCrop(output_size=args.crop_size, scale=(0.8, 1.0), randomizer=random_generator),  # TODO: Random rotation?
        #utils.RandomHorizontalFlip(p=0.5, randomizer=random_generator)
        # TODO: Color jitter?
    ]
    train_dataset = utils.SceneflowDataset(args.sceneflow_root, string_exclude='TEST', transforms=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last = False)
    
    val_transforms = [utils.ToTensor(), utils.Rescale()]
    val_dataset = utils.SceneflowDataset(args.sceneflow_root, string_include='TEST', transforms=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last = False)
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger(save_dir=Path.cwd(), name="lightning_logs")
    trainer = pl.Trainer(gpus=args.num_gpus, min_epochs=args.min_epochs, max_epochs=args.max_epochs, logger=logger, callbacks=[lr_monitor], checkpoint_callback=True)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print('stall')

if __name__ == "__main__":
    main()
