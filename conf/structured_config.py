from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
import torch


@dataclass
class Train:
    batch_size: int = 16
    epoch_count: int = 25
    lr: float = 3e-3
    # liner sheduler for lr
    start_factor: float = 1.0
    end_factor: float = 1

    # weights initialize params, uniform distribution
    # from 0 to param
    pi = 2 * torch.acos(torch.zeros(1)).item()
    conv_weight: float = 1
    bias: float = 0.01
    quantum_weight: float = pi / 2

@dataclass
class Data:
    # path to save dataset
    load_dir: str = "/mnist"
    # width ang heigth after resizing 1x28x28 pics
    width: int = 14
    height: int = 14
    # pics quantity of each class
    min_length: int = 40

@dataclass
class Plot:
    path: str = "graphics.png"


@dataclass
class Config:
    train: Train = field(default_factory=Train)
    data: Data = field(default_factory=Data)
    plot: Plot = field(default_factory=Plot)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
