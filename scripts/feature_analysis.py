#!/usr/bin/env python3
"""
Standalone feature importance analysis script
Run this to analyze feature importance without running the full testing pipeline
"""

import os
import sys
import logging
from datetime import datetime

import torch
import numpy as np

from psco.config import Config
from psco.data_processor import DataProcessor
from psco.model import create_model
from psco.trainer import load_model
from psco.feature_importance import (
    analyze_feature_importance,
    compare_feature_importance_methods,
)


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main(model_path: str, processor_path: str, methods=None, sample_size=1000):
    """
    Run standalone feature importance analysis

    Args:
        model_path: Path to the trained model
        processor_path: Path to the data processor
        methods: List of methods to use
        sample_size: Size of test sample to use for analysis
    """
    if methods is None:
        methods = ["permutation", "shap"]

    setup_logging()
    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # 1. Load data
        logging.info("Loading and processing data...")
        data_processor = DataProcessor(config)
        df = data_processor.load_data()
        df_test = data_processor.preprocess_data(df, data_type="test")

        # Load saved processor
        if os.path.exists(processor_path):
            logging.info("Loading saved data processor...")
            processor_state = torch.load(processor_path, weights_only=False)
            data_processor.scaler = processor_state["scaler"]
            data_processor.class_weights = processor_state["class_weights"]
        else:
            raise FileNotFoundError(f"Processor file not found: {processor_path}")

        # Transform test data
        X_test, y_test = data_processor.transform_test_data(df_test)

        # Use a sample for faster analysis
        if sample_size < len(X_test):
            logging.info(
                f"Using random sample of {sample_size} from {len(X_test)} test samples"
            )
            indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]

        # 2. Load model
        logging.info("Loading model...")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=False)
            if "config" in checkpoint:
                model = create_model(checkpoint["config"])
            else:
                model = create_model(config)

            epoch, metrics = load_model(model, model_path)
            logging.info(f"Model loaded, epoch: {epoch}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # 3. Feature importance analysis
        logging.info("Starting feature importance analysis...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_names = config.data.feature_columns

        importance_results = analyze_feature_importance(
            model=model,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            device=device,
            save_dir=config.paths.plots_dir,
            timestamp=timestamp,
            methods=methods,
        )

        # 4. Generate comparison analysis
        logging.info("Generating method comparison...")
        comparison_df = compare_feature_importance_methods(importance_results)

        # Save comparison
        comparison_path = os.path.join(
            config.paths.plots_dir, f"feature_importance_comparison_{timestamp}.csv"
        )
        comparison_df.to_csv(comparison_path, index=False)
        logging.info(f"Feature importance comparison saved to: {comparison_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS SUMMARY")
        print("=" * 60)

        print("\\nTop 10 Most Important Features (averaged across methods):")
        print("-" * 50)
        for i, row in comparison_df.head(10).iterrows():
            print(
                f"{int(row['Rank_Mean']):2d}. {row['Feature']:20s} (Avg Rank: {row['Rank_Mean']:.1f})"
            )

        print("\\nMost Consistent Features (lowest rank variation):")
        print("-" * 50)
        consistent = comparison_df.nsmallest(5, "Rank_StdDev")
        for i, row in consistent.iterrows():
            print(f"{row['Feature']:20s} (Rank StdDev: {row['Rank_StdDev']:.2f})")

        print("\\nAnalysis complete! Check the plots directory for visualizations.")

        return importance_results, comparison_df

    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        import traceback

        logging.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone PSCO feature importance analysis"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--processor_path", type=str, required=True, help="Path to data processor"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["permutation", "shap"],
        default=["permutation", "shap"],
        help="Feature importance methods to use",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="Size of test sample to use for analysis (default: 10000)",
    )

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        processor_path=args.processor_path,
        methods=args.methods,
        sample_size=args.sample_size,
    )
