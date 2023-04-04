"""
Script to instantiate a StereoNet model + train on the SceneFlow dataset.
"""

import hydra

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from stereonet.model import StereoNet
import stereonet.utils as utils
import stereonet.types_hydra as th


@hydra.main(version_base=None)
def main(cfg: th.StereoNetConfig) -> int:

    model_config = hydra.utils.instantiate(cfg.model)
    training_data_config = [hydra.utils.instantiate(data_cfg) for data_cfg in cfg.training.data]
    validation_data_config = [hydra.utils.instantiate(data_cfg) for data_cfg in cfg.validation.data]

    # Instantiate model with built in optimizer
    if isinstance(model_config, th.CheckpointModel):
        model = StereoNet.load_from_checkpoint(model_config.model_checkpoint_path)
    elif isinstance(model_config, th.StereoNetModel):
        model = StereoNet(k_downsampling_layers=model_config.k_downsampling_layers,
                          k_refinement_layers=model_config.k_refinement_layers,
                          candidate_disparities=model_config.candidate_disparities,
                          mask=False,
                          optimizer_config=cfg.training.optimizer,
                          scheduler_config=cfg.training.scheduler)

    # # Get datasets
    train_loader = utils.construct_dataloaders(data_config=training_data_config,
                                               loader_cfg=cfg.loader,
                                               is_training=True,
                                               shuffle=True, num_workers=8, drop_last=False)

    val_loader = utils.construct_dataloaders(data_config=validation_data_config,
                                             loader_cfg=cfg.loader,
                                             is_training=False,
                                             shuffle=False, num_workers=8, drop_last=False)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss_epoch', save_top_k=-1, mode='min')

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger(save_dir=cfg.logging.lightning_log_root, name="lightning_logs")
    trainer = pl.Trainer(devices=cfg.global_settings.devices.num,
                         accelerator=cfg.global_settings.devices.type,
                         min_epochs=cfg.training.min_epochs,
                         max_epochs=cfg.training.max_epochs,
                         logger=logger,
                         callbacks=[lr_monitor, checkpoint_callback])

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
