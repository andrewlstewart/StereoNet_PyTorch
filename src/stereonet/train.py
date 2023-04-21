"""
Script to instantiate a StereoNet model + train on the SceneFlow or KeystoneDepth dataset.
"""

from typing import Optional

import hydra

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from stereonet.model import StereoNet
import stereonet.datasets as std
import stereonet.types as stt


@hydra.main(version_base=None, config_name="config")
def main(cfg: stt.StereoNetConfig) -> int:
    config: stt.StereoNetConfig = hydra.utils.instantiate(cfg, _convert_="all")['stereonet_config']

    if config.training is None:
        raise Exception("Need to provide training arguments to train the model.")

    if config.training.random_seed is not None:
        pl.seed_everything(config.training.random_seed)

    checkpoint_path: Optional[str] = None
    # Instantiate model with built in optimizer
    if isinstance(config.model, stt.CheckpointModel):
        checkpoint_path = config.model.model_checkpoint_path
        model = StereoNet.load_from_checkpoint(checkpoint_path)
    elif isinstance(config.model, stt.StereoNetModel):
        model = StereoNet(in_channels=config.model.in_channels,
                          k_downsampling_layers=config.model.k_downsampling_layers,
                          k_refinement_layers=config.model.k_refinement_layers,
                          candidate_disparities=config.model.candidate_disparities,
                          mask=config.training.mask,
                          optimizer_partial=config.training.optimizer_partial,
                          scheduler_partial=config.training.scheduler_partial)
    else:
        raise Exception("Unknown model type")

    # Get datasets
    train_loader = std.construct_dataloaders(data_config=config.training,
                                             is_training=True,
                                             shuffle=True, num_workers=8, drop_last=False)

    val_loader = None
    if config.validation is not None:
        val_loader = std.construct_dataloaders(data_config=config.validation,
                                               is_training=False,
                                               shuffle=False, num_workers=8, drop_last=False)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss_epoch', save_top_k=-1, mode='min')

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger(save_dir=config.logging.lightning_log_root, name="lightning_logs")
    trainer = pl.Trainer(devices=config.global_settings.devices.num,
                         accelerator=config.global_settings.devices.type,
                         min_epochs=config.training.min_epochs,
                         max_epochs=config.training.max_epochs,
                         logger=logger,
                         callbacks=[lr_monitor, checkpoint_callback],
                         deterministic=config.training.deterministic)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=checkpoint_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
