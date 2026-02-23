"""
Neural network model architecture for PSCO classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Union

from .config import Config


class ResidualBlock(nn.Module):
    """Residual connection block with layer normalization"""

    def __init__(self, size: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)

    def forward(self, x):
        # First sublayer
        residual = x
        x = self.norm1(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = x + residual

        # Second sublayer
        residual = x
        x = self.norm2(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = x + residual

        return x


class PSCOModel(nn.Module):
    """PSCO classification model with attention and residual connections"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Model configuration
        input_size = config.model.input_size
        hidden_sizes = config.model.hidden_sizes
        num_classes = config.model.num_classes
        dropout = config.model.dropout

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_dropout = nn.Dropout(dropout)

        # Hidden layers with spectral normalization
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            layer = spectral_norm(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.hidden_layers.append(layer)

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for size in hidden_sizes:
            self.residual_blocks.append(ResidualBlock(size, dropout))

        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for size in hidden_sizes:
            self.batch_norms.append(nn.BatchNorm1d(size))

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Kaiming normal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the neural network

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Input layer
        x = self.input_layer(x)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        x = self.input_dropout(x)

        # Hidden layers with residual connections
        for i, (layer, residual_block, bn) in enumerate(
            zip(self.hidden_layers, self.residual_blocks[1:], self.batch_norms[1:])
        ):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

            # Add residual connection
            x = residual_block(x)

        # Output layer
        x = self.output_layer(x)

        return x

    def get_model_info(self):
        """Get model information including parameter counts"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": total_params * 4 / 1024 / 1024,  # Assume float32
            "input_size": self.config.model.input_size,
            "hidden_sizes": self.config.model.hidden_sizes,
            "num_classes": self.config.model.num_classes,
            "dropout": self.config.model.dropout,
        }


class ModelEnsemble(nn.Module):
    """Model ensemble for improved prediction accuracy"""

    def __init__(self, models: list):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # Average ensemble
        ensemble_output = torch.stack(outputs, dim=0).mean(dim=0)
        return ensemble_output

    def predict_with_uncertainty(self, x):
        """Predict and calculate uncertainty"""
        outputs = []
        for model in self.models:
            outputs.append(F.softmax(model(x), dim=1))

        # Calculate average probability
        probs = torch.stack(outputs, dim=0)
        mean_probs = probs.mean(dim=0)

        # Calculate uncertainty (standard deviation)
        std_probs = probs.std(dim=0)

        return mean_probs, std_probs


def create_model(
    config: Config
) -> Union[PSCOModel, nn.DataParallel]:
    """
    Create model based on configuration

    Args:
        config: Configuration object
    """
    model = PSCOModel(config)

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model
