"""
Data processing utilities for PSCO model
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Dict, Optional
import os

from .config import Config


class DataProcessor:
    """Data processor for loading, preprocessing, and splitting data"""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.class_weights = None

    def load_data(self) -> pd.DataFrame:
        """Load dataset from CSV file"""
        if not os.path.exists(self.config.data.dataset_path):
            raise FileNotFoundError(
                f"Data file not found: {self.config.data.dataset_path}"
            )

        df = pd.read_csv(self.config.data.dataset_path)
        logging.info(f"Loaded {len(df)} records")
        return df

    def preprocess_data(
        self, df: pd.DataFrame, data_type: str = "train"
    ) -> pd.DataFrame:
        """
        Preprocess data based on type (train or test)

        Args:
            df: Input DataFrame containing raw data
            data_type: Either "train" or "test"

        Returns:
            Processed DataFrame
        """
        logging.info(f"Processing {data_type} data...")

        # Filter based on data type
        if data_type == "train":
            df_filtered = df[df["inspect"] == 1].copy()
        else:
            df_filtered = df[df["inspect"] == 0].copy()

        logging.info(f"{data_type} data size: {len(df_filtered)}")

        # Encode labels
        df_processed = self._encode_labels(df_filtered)

        # Select features and target
        columns_to_keep = self.config.data.feature_columns + [
            self.config.data.target_column
        ]
        return df_processed[columns_to_keep]

    def _encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode labels based on qualified and detained status"""
        conditions = [
            (df["qualified"] == 1) & (df["detained"] == 0),  # qualified & not detained
            (df["qualified"] == 0)
            & (df["detained"] == 0),  # not qualified & not detained
            (df["qualified"] == 0) & (df["detained"] == 1),  # not qualified & detained
        ]
        choices = [0, 1, 2]
        df["label"] = np.select(conditions, choices, default=-1)

        invalid_labels = df[df["label"] == -1]
        if len(invalid_labels) > 0:
            logging.warning(
                f"Found {len(invalid_labels)} invalid labels, removing them"
            )
            df = df[df["label"] != -1]

        label_dist = df["label"].value_counts().to_dict()
        logging.info(f"Label distribution: {label_dist}")

        return df

    def split_data(
        self, df: pd.DataFrame, test_size: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets

        Args:
            df: Processed DataFrame
            test_size: Test set ratio

        Returns:
            X_train, X_test, y_train, y_test
        """
        if test_size is None:
            test_size = self.config.training.val_split

        X = df[self.config.data.feature_columns].to_numpy()
        y = df[self.config.data.target_column].to_numpy()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=self.config.training.random_state,
            stratify=y,
            shuffle=True,
        )

        logging.info(
            f"Training set: {len(X_train)} ({len(X_train) / (len(X_train) + len(X_test)) * 100:.1f}%)"
        )
        logging.info(
            f"Validation set: {len(X_test)} ({len(X_test) / (len(X_train) + len(X_test)) * 100:.1f}%)"
        )

        return X_train, X_test, y_train, y_test

    def transform_test_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform test data using the fitted scaler

        Args:
            df: Test DataFrame

        Returns:
            X_test, y_test
        """
        X = df[self.config.data.feature_columns].to_numpy()
        y = df[self.config.data.target_column].to_numpy()

        X_scaled = self.scaler.transform(X)

        return X_scaled, y

    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced dataset"""
        class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)

        self.class_weights = {i: class_weights[i] for i in range(len(class_weights))}
        logging.info(f"Class weights: {self.class_weights}")

        return self.class_weights

    def get_class_names(self) -> list:
        """Return class names for visualization"""
        return [
            "Qualified & Undetained",
            "Not Qualified & Undetained",
            "Not Qualified & Detained",
        ]

    def log_data_distribution(self, y: np.ndarray, data_type: str = "data"):
        """Log data distribution by class"""
        class_names = self.get_class_names()
        unique, counts = np.unique(y, return_counts=True)

        logging.info(f"{data_type} label distribution:")
        for i, count in zip(unique, counts):
            percentage = count / len(y) * 100
            logging.info(f"  Class {i} ({class_names[i]}): {count} ({percentage:.1f}%)")
