from typing import Optional, List, Any, Callable

from abc import ABC
from dataclasses import dataclass

import torch.nn
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
    in_channels: int
    k_downsampling_layers: int
    k_refinement_layers: int
    candidate_disparities: int


@dataclass
class Loader:
    batch_size: int


# https://stackoverflow.com/a/69822584
class Data(ABC):
    def __init__(self, root_path: str,
                 max_size: Optional[List[int]] = None,
                 transforms: Optional[List[torch.nn.Module]] = None):
        self.root_path = root_path
        self.max_size = max_size
        self.transforms = transforms


class SceneflowData(Data):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class KeystoneDepthData(Data):
    def __init__(self, split_ratio: float, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.split_ratio = split_ratio


@dataclass
class DataDebug:
    enabled: bool
    limit_train_batches: int


@dataclass
class Training:
    min_epochs: int
    max_epochs: int
    mask: bool
    data: List[Data]
    loader: Loader
    debug: DataDebug
    optimizer_partial: Callable[[torch.nn.Module], torch.optim.Optimizer]
    scheduler_partial: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None
    random_seed: Optional[int] = None
    deterministic: Optional[bool] = None
    fast_dev_run: bool = False


@dataclass
class Validation:
    data: List[Data]
    loader: Loader


@dataclass
class StereoNetConfig:
    global_settings: GlobalSettings
    logging: Logging
    model: Model
    training: Optional[Training] = None
    validation: Optional[Validation] = None
