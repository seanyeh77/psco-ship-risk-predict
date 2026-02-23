"""
Training utilities for PSCO model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import time
import os
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config


class PSCODataset(Dataset):
    """PSCO dataset wrapper"""

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


class EarlyStopping:
    """Early stopping callback"""

    def __init__(
        self, patience: int = 7, min_delta: float = 0, restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        """Save best model weights"""
        self.best_weights = model.state_dict().copy()


class PSCOTrainer:
    """PSCO model trainer"""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        class_weights: Optional[Dict[int, float]] = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.model: nn.Module = model
        self.config: Config = config
        self.device: torch.device = device
        self.class_weights: Dict[int, float] | None = class_weights

        self.model.to(device)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            eps=1e-8,
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.learning_rate * 0.01,
        )

        # Loss function - ensure weights are ordered by class index
        if class_weights:
            sorted_weights = [class_weights[i] for i in sorted(class_weights.keys())]
            weight_tensor = torch.tensor(
                sorted_weights, dtype=torch.float32, device=self.device
            )
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Mixed precision training
        self.scaler = torch.GradScaler()

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=0.001,
            restore_best_weights=True,
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                output = self.model(data)
                loss = self.criterion(output, target)

            # Mixed precision backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Complete training process"""
        logging.info(f"Starting training on device: {self.device}")
        logging.info(
            f"Training params: epochs={self.config.training.epochs}, lr={self.config.training.learning_rate}"
        )

        start_time = time.time()

        for epoch in range(self.config.training.epochs):
            epoch_start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)

            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(current_lr)

            epoch_time = time.time() - epoch_start_time

            # Logging
            logging.info(
                f"Epoch {epoch + 1}/{self.config.training.epochs} "
                f"({epoch_time:.2f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"LR: {current_lr:.6f}"
            )

            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time:.2f}s")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "learning_rates": self.learning_rates,
            "total_time": total_time,
            "final_train_acc": self.train_accuracies[-1],
            "final_val_acc": self.val_accuracies[-1],
        }

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    output = self.model(data)
                    probabilities = F.softmax(output, dim=1)

                predictions = output.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets)) * 100

        return {
            "accuracy": accuracy,
            "predictions": all_predictions,
            "targets": all_targets,
            "probabilities": all_probabilities,
        }

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.train_losses, label="Train Loss", color="blue")
        axes[0, 0].plot(self.val_losses, label="Validation Loss", color="red")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(self.train_accuracies, label="Train Accuracy", color="blue")
        axes[0, 1].plot(self.val_accuracies, label="Validation Accuracy", color="red")
        axes[0, 1].set_title("Training and Validation Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate schedule
        axes[1, 0].plot(self.learning_rates, label="Learning Rate", color="green")
        axes[1, 0].set_title("Learning Rate Schedule")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale("log")

        # Training statistics
        axes[1, 1].axis("off")
        stats_text = f"""
        Training Statistics:
        
        Final Train Accuracy: {self.train_accuracies[-1]:.2f}%
        Final Val Accuracy: {self.val_accuracies[-1]:.2f}%
        Best Val Accuracy: {max(self.val_accuracies):.2f}%
        
        Final Train Loss: {self.train_losses[-1]:.4f}
        Final Val Loss: {self.val_losses[-1]:.4f}
        Best Val Loss: {min(self.val_losses):.4f}
        
        Total Epochs: {len(self.train_losses)}
        """
        axes[1, 1].text(
            0.1,
            0.9,
            stats_text,
            transform=axes[1, 1].transAxes,
            verticalalignment="top",
            fontsize=12,
            family="monospace",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"Training history plot saved to: {save_path}")

        # plt.show()


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation"""
    train_dataset = PSCODataset(X_train, y_train)
    val_dataset = PSCODataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    show_metrics: bool = True,
):
    """
    Plot confusion matrix with detailed metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
        show_metrics: Whether to show detailed metrics
    """
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    cm = confusion_matrix(y_true, y_pred)

    # Calculate metrics
    metrics = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )
    precision, recall, f1, support = metrics

    # Ensure arrays are numpy arrays
    precision = np.array(precision)
    recall = np.array(recall)
    f1 = np.array(f1)
    support = np.array(support)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Create shorter class names for display
    short_class_names = []
    for name in class_names:
        if len(name) > 15:
            # Split long names into multiple lines
            words = name.split()
            if (
                len(words) == 4 and words[2] == "&"
            ):  # For "Not Qualified & Undetained/Detained"
                short_name = f"{words[0]} {words[1]}\n& {words[3]}"
            elif len(words) == 3 and words[1] == "&":  # For "Qualified & Undetained"
                short_name = f"{words[0]}\n& {words[2]}"
            else:
                short_name = name[:15] + "..."
        else:
            short_name = name
        short_class_names.append(short_name)

    # Plot 1: Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_class_names,
        yticklabels=short_class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title(
        f"Confusion Matrix (Accuracy: {accuracy:.4f})", fontsize=16, fontweight="bold"
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Save confusion matrix
    if save_path:
        cm_path = save_path.replace(".png", "_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        logging.info(f"Confusion matrix saved to: {cm_path}")

    plt.tight_layout()
    # plt.show()

    # Plot 2: Metrics Table (if requested)
    if show_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")

        # Create metrics table
        metrics_data = []
        for i, class_name in enumerate(class_names):
            metrics_data.append(
                [
                    class_name,
                    f"{precision[i]:.4f}",
                    f"{recall[i]:.4f}",
                    f"{f1[i]:.4f}",
                    f"{support[i]}",
                ]
            )

        # Add average metrics
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        total_support = np.sum(support)

        metrics_data.append(
            [
                "Average",
                f"{avg_precision:.4f}",
                f"{avg_recall:.4f}",
                f"{avg_f1:.4f}",
                f"{total_support}",
            ]
        )

        # Create table
        table = ax.table(
            cellText=metrics_data,
            colLabels=["Class", "Precision", "Recall", "F1-Score", "Support"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.0)

        # Style the header
        for i in range(5):  # 5 columns
            if (0, i) in table.get_celld():
                cell = table.get_celld()[(0, i)]
                cell.set_facecolor("#40466e")
                cell.set_text_props(weight="bold", color="white")

        # Style the average row
        for i in range(5):  # 5 columns
            if (len(class_names) + 1, i) in table.get_celld():
                cell = table.get_celld()[(len(class_names) + 1, i)]
                cell.set_facecolor("#f0f0f0")
                cell.set_text_props(weight="bold")

        # Style class name cells to handle long text
        for i in range(len(class_names)):
            if (i + 1, 0) in table.get_celld():
                cell = table.get_celld()[(i + 1, 0)]
                cell.set_text_props(wrap=True)

        ax.set_title("Classification Metrics", fontsize=16, fontweight="bold", pad=20)

        # Save metrics table
        if save_path:
            metrics_path = save_path.replace(".png", "_metrics.png")
            plt.savefig(metrics_path, dpi=300, bbox_inches="tight")
            logging.info(f"Metrics table saved to: {metrics_path}")

        plt.tight_layout()
        # plt.show()

    # Log detailed metrics
    if show_metrics:
        logging.info("=== Classification Metrics ===")
        logging.info(f"Overall Accuracy: {accuracy:.4f}")
        logging.info(f"Average Precision: {avg_precision:.4f}")
        logging.info(f"Average Recall: {avg_recall:.4f}")
        logging.info(f"Average F1-Score: {avg_f1:.4f}")
        logging.info("\nPer-class metrics:")
        for i, class_name in enumerate(class_names):
            logging.info(
                f"  {class_name}: Precision={precision[i]:.4f}, "
                f"Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}"
            )

    # Overall save path (backward compatibility)
    if save_path and not show_metrics:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Confusion matrix saved to: {save_path}")


def save_model(
    model: nn.Module,
    config: Config,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[CosineAnnealingLR] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict] = None,
):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Handle DataParallel models
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        "model_state_dict": model_state_dict,
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)
    logging.info(f"Model saved to: {save_path}")


def load_model(
    model: nn.Module,
    load_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    """Load model checkpoint"""
    checkpoint = torch.load(load_path, weights_only=False)

    # Load model weights
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logging.info(f"Model loaded from {load_path}")

    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
