"""
"""

from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl

from src.model import StereoNet
import src.utils as utils

def main():

    # Instantiate model with built in optimizer
    model = StereoNet()

    # Get datasets
    sceneflow_path = Path(r'C:\Users\andre\Documents\Python\data\SceneFlow')
    random_generator = np.random.default_rng(seed=42)
    train_transforms = [
        utils.ToTensor(),
        utils.RandomResizedCrop(output_size=(540//2, 960//2), scale=(0.8, 1.0), randomizer=random_generator),  # TODO: Random rotation?
        utils.RandomHorizontalFlip(p=0.5, randomizer=random_generator)
        # TODO: Color jitter?
    ]
    train_dataset = utils.SceneflowDataset(sceneflow_path, string_exclude='TEST', transforms=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8, drop_last = False)
    
    val_transforms = [utils.ToTensor()]
    val_dataset = utils.SceneflowDataset(sceneflow_path, string_include='TEST', transforms=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=8, drop_last = False)
    
    trainer = pl.Trainer(gpus=1)

    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()
