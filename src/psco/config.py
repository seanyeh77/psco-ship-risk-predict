"""
Configuration settings for PSCO model
"""

import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    hidden_sizes: list[int] = field(default_factory=lambda: [32, 64, 128, 64, 32])
    input_size: int = 22  # number of features
    num_classes: int = 3  # number of classes
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration"""

    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-3
    epochs: int = 100
    patience: int = 17
    min_delta: float = 0.001
    val_split: float = 0.1
    random_state: int = 42
    num_workers: int = 4


@dataclass
class DataConfig:
    """Data configuration"""

    dataset_path: str = "data/simulated/simulated_data.csv"
    feature_columns: list[str] = field(
        default_factory=lambda: [
            "YOB",
            "InitialInsp",
            "DetInsp",
            "FollowUpInsp",
            "StandInsp",
            "NoInsp",
            "NoDef",
            "ISM",
            "MARPOL",
            "CertDoc",
            "PAMach",
            "SafNav",
            "RadioCom",
            "EmergSyst",
            "FireSafety",
            "MLC",
            "Alarms",
            "ISPS",
            "OTDef",
            "WWTightCond",
            "LifeApl",
            "GT",
        ]
    )
    target_column: str = "label"


@dataclass
class PathConfig:
    """Path configuration"""

    models_dir: str = "models"
    plots_dir: str = "outputs/plots"
    logs_dir: str = "logs"

    def __post_init__(self):
        # Ensure directories exist
        for dir_path in [self.models_dir, self.plots_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class Config:
    """Main configuration class"""

    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig())
    data: DataConfig = field(default_factory=lambda: DataConfig())
    paths: PathConfig = field(default_factory=lambda: PathConfig())


# Default configuration instance
default_config = Config()
