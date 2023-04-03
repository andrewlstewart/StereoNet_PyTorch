from typing import List, Union, Optional

from dataclasses import dataclass


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
class Model:
    k_downsampling_layers: int
    k_refinement_layers: int
    candidate_disparities: int


@dataclass
class Loader:
    batch_size: int


@dataclass
class RMSprop:
    learning_rate: float
    weight_decay: float


@dataclass
class Optimizer:
    name: str
    properties: Optional[RMSprop] = None


@dataclass
class ExponentialLR:
    gamma: float


@dataclass
class Scheduler:
    name: str
    properties: Optional[ExponentialLR] = None


@dataclass
class CenterCropProperties:
    scale: float


@dataclass
class Transform:
    name: str
    properties: Optional[CenterCropProperties] = None


@dataclass
class SceneflowProperties:
    root_path: str
    transforms: List[Transform]


@dataclass
class KeystoneDepthProperties:
    root_path: str
    split_ratio: float
    transforms: List[Transform]


@dataclass
class Data:
    name: str
    properties: Union[KeystoneDepthProperties, SceneflowProperties]


@dataclass
class DataDebug:
    enabled: bool
    limit_train_batches: int


@dataclass
class Training:
    min_epochs: int
    max_epochs: int
    optimier: Optimizer
    scheduler: Scheduler
    data: List[Data]
    debug: DataDebug


@dataclass
class Validation:
    data: Data


@dataclass
class StereoNetConfig:
    hydra: Hydra
    global_settings: GlobalSettings
    logging: Logging
    model: Model
    loader: Loader
    training: Training
    validation: Validation
