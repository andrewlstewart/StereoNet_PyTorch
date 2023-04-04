from typing import List, Optional

import stereonet.types_stereonet as ts

from abc import ABC
from dataclasses import dataclass

import torch.optim


@dataclass
class Run:
    dir: str


@dataclass
class Hydra:
    run: Run


@dataclass
class Devices:
    num: int
    type: str


@dataclass
class GlobalSettings:
    random_seed: int
    devices: Devices


@dataclass
class Logging:
    lightning_log_root: str


@dataclass
class Model(ABC):
    ...


@dataclass
class CheckpointModel(Model):
    model_checkpoint_path: str


@dataclass
class StereoNetModel(Model):
    k_downsampling_layers: int
    k_refinement_layers: int
    candidate_disparities: int


@dataclass
class Loader:
    batch_size: int


@dataclass
class SceneflowProperties:
    root_path: str
    transforms: List[ts.TorchTransformer]


@dataclass
class KeystoneDepthProperties:
    root_path: str
    split_ratio: float
    transforms: List[ts.TorchTransformer]


@dataclass
class Data(ABC):
    root_path: str
    transforms: List[ts.TorchTransformer]


@dataclass
class SceneflowData(Data):
    pass


@dataclass
class KeystoneDepthData(Data):
    split_ratio: float


@dataclass
class DataDebug:
    enabled: bool
    limit_train_batches: int


@dataclass
class Training:
    min_epochs: int
    max_epochs: int
    data: List[Data]
    debug: DataDebug
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None


@dataclass
class Validation:
    data: List[Data]


@dataclass
class StereoNetConfig:
    hydra: Hydra
    global_settings: GlobalSettings
    logging: Logging
    model: Model
    loader: Loader
    training: Training
    validation: Validation
