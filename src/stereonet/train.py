"""
Script to instantiate a StereoNet model + train on the SceneFlow dataset.
"""

import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from stereonet.model import StereoNet
import stereonet.utils as utils


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> int:

    # Instantiate model with built in optimizer
    if 'model_checkpoint_path' in cfg.training:
        model = StereoNet.load_from_checkpoint(cfg.model.model_checkpoint_path)
    else:
        model = StereoNet(k_downsampling_layers=cfg.model.k_downsampling_layers, k_refinement_layers=cfg.model.k_refinement_layers, candidate_disparities=cfg.model.candidate_disparities, mask=False)

    # # Get datasets
    train_loader = utils.construct_dataloaders(data_cfg=cfg.training.data,
                                               loader_cfg=cfg.loader,
                                               is_training=True,
                                               shuffle=True, num_workers=8, drop_last=False)

    val_loader = utils.construct_dataloaders(data_cfg=cfg.validation.data,
                                             loader_cfg=cfg.loader,
                                             is_training=False,
                                             shuffle=False, num_workers=8, drop_last=False)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss_epoch', save_top_k=-1, mode='min')

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger(save_dir=cfg.logging.lightning_log_root, name="lightning_logs")
    trainer = pl.Trainer(gpus=cfg.global_settings.num_gpus,
                         min_epochs=cfg.training.min_epochs,
                         max_epochs=cfg.training.max_epochs,
                         logger=logger,
                         callbacks=[lr_monitor, checkpoint_callback])

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
