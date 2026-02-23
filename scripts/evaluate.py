"""
Model testing script - evaluate trained model on test data (inspect==0)
Includes comprehensive feature importance analysis using multiple methods:
- Permutation importance
- Input gradient importance
- Integrated gradients importance
"""

import os
import sys
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional
from datetime import datetime

# Add src directory to path for psco package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
sys.path.insert(0, src_dir)

from psco.config import Config
from psco.data_processor import DataProcessor
from psco.model import create_model
from psco.trainer import plot_confusion_matrix, load_model
from psco.feature_importance import (
    analyze_feature_importance,
    compare_feature_importance_methods,
)
from sklearn.metrics import classification_report


def setup_logging(config: Config):
    """Setup logging configuration"""
    os.makedirs(config.paths.logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.paths.logs_dir, f"testing_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    return log_file


def plot_class_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: Optional[str] = None,
):
    """Plot class distribution comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # True label distribution
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    axes[0].bar(
        [class_names[i] for i in unique_true],
        counts_true,
        color=["skyblue", "lightcoral", "lightgreen"],
    )
    axes[0].set_title("True Label Distribution")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    # Predicted label distribution
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    axes[1].bar(
        [class_names[i] for i in unique_pred],
        counts_pred,
        color=["skyblue", "lightcoral", "lightgreen"],
    )
    axes[1].set_title("Predicted Label Distribution")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Class distribution plot saved to: {save_path}")

    plt.show()


def main(
    model_path: Optional[str] = None,
    processor_path: Optional[str] = None,
    methods: Optional[list] = None,
    sample_size: int = 1000,
):
    """Main testing pipeline"""
    if methods is None:
        methods = ["permutation", "shap"]

    config = Config()

    log_file = setup_logging(config)
    logging.info("=" * 50)
    logging.info("PSCO Model Testing")
    logging.info("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # 1. Data processing
        logging.info("1. Processing data...")
        data_processor = DataProcessor(config)

        df = data_processor.load_data()

        # Process test data (inspect==0)
        df_test = data_processor.preprocess_data(df, data_type="test")
        logging.info(f"Test data shape: {df_test.shape}")

        # Load saved processor if available
        if processor_path and os.path.exists(processor_path):
            logging.info("Loading saved data processor...")
            processor_state = torch.load(processor_path, weights_only=False)
            data_processor.scaler = processor_state["scaler"]
            data_processor.class_weights = processor_state["class_weights"]
        else:
            # Retrain scaler using training data
            logging.info("Processor not found, retraining scaler with training data...")
            df_train = data_processor.preprocess_data(df, data_type="train")
            X_train, _, y_train, _ = data_processor.split_data(df_train)

        # Transform test data
        X_test, y_test = data_processor.transform_test_data(df_test)

        # Use a sample for feature importance analysis if specified
        if sample_size < len(X_test):
            logging.info(
                f"Using random sample of {sample_size} from {len(X_test)} test samples for analysis"
            )
            sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_test_sample = X_test[sample_indices]
            y_test_sample = y_test[sample_indices]
        else:
            X_test_sample = X_test
            y_test_sample = y_test

        data_processor.log_data_distribution(y_test, "Test set")

        # 2. Load model
        logging.info("2. Loading model...")

        # Load checkpoint to get configuration
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=False)
            if "config" in checkpoint:
                saved_config = checkpoint["config"]
                model = create_model(saved_config)
                logging.info("Using saved configuration")
            else:
                model = create_model(config)
                logging.info("Using current configuration")

            epoch, metrics = load_model(model, model_path)
            logging.info(f"Model loaded, epoch: {epoch}")
            if metrics:
                logging.info(
                    f"Final validation accuracy: {metrics.get('final_val_acc', 'N/A')}"
                )
        else:
            raise FileNotFoundError(
                "Model file not found, please train the model first"
            )

        # 3. Testing
        logging.info("3. Running inference...")
        model.eval()

        X_test_tensor = torch.FloatTensor(X_test).to(device)

        with torch.no_grad():
            # Batch prediction to avoid memory issues
            batch_size = config.training.batch_size
            all_predictions = []
            all_probabilities = []

            for i in range(0, len(X_test_tensor), batch_size):
                batch_X = X_test_tensor[i : i + batch_size]
                batch_output = model(batch_X)
                batch_probs = torch.nn.functional.softmax(batch_output, dim=1)
                batch_preds = batch_output.argmax(dim=1)

                all_predictions.extend(batch_preds.cpu().numpy())
                all_probabilities.extend(batch_probs.cpu().numpy())

        # 4. Calculate metrics
        logging.info("4. Computing metrics...")
        accuracy = np.mean(np.array(all_predictions) == y_test) * 100
        logging.info(f"Test accuracy: {accuracy:.2f}%")

        # 5. Classification report
        class_names = data_processor.get_class_names()
        report = classification_report(
            y_test, all_predictions, target_names=class_names, digits=4
        )
        logging.info(f"Classification report:\\n{report}")

        # 6. Generate visualizations
        logging.info("6. Generating visualizations...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Confusion matrix
        cm_path = os.path.join(
            config.paths.plots_dir, f"test_confusion_matrix_{timestamp}.png"
        )
        plot_confusion_matrix(y_test, np.array(all_predictions), class_names, cm_path)

        # Class distribution
        dist_path = os.path.join(
            config.paths.plots_dir, f"test_class_distribution_{timestamp}.png"
        )
        plot_class_distribution(
            y_test, np.array(all_predictions), class_names, dist_path
        )

        # 7. Feature Importance Analysis
        logging.info("7. Performing feature importance analysis...")
        feature_names = config.data.feature_columns

        importance_results = analyze_feature_importance(
            model=model,
            X_test=X_test_sample,
            y_test=y_test_sample,
            feature_names=feature_names,
            device=device,
            save_dir=config.paths.plots_dir,
            timestamp=timestamp,
            methods=methods,
        )

        # 8. Generate comparison analysis
        logging.info("8. Generating method comparison...")
        comparison_df = compare_feature_importance_methods(importance_results)

        # Save comparison
        comparison_path = os.path.join(
            config.paths.plots_dir, f"feature_importance_comparison_{timestamp}.csv"
        )
        comparison_df.to_csv(comparison_path, index=False)
        logging.info(f"Feature importance comparison saved to: {comparison_path}")

        logging.info("=" * 50)
        logging.info("Testing completed!")
        logging.info("=" * 50)
        logging.info(f"Test accuracy: {accuracy:.2f}%")
        logging.info(f"Confusion matrix: {cm_path}")
        logging.info(f"Class distribution: {dist_path}")
        logging.info(
            f"Feature importance: {os.path.join(config.paths.plots_dir, f'feature_importance_{timestamp}.png')}"
        )
        logging.info(f"Feature importance comparison: {comparison_path}")
        logging.info(f"Log file: {log_file}")

        # Print summary similar to run_feature_importance.py
        print("\\n" + "=" * 60)
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

        return {
            "accuracy": accuracy,
            "predictions": all_predictions,
            "probabilities": all_probabilities,
            "y_true": y_test,
            "report": report,
            "feature_importance": importance_results,
            "comparison_df": comparison_df,
        }

    except Exception as e:
        logging.error(f"Testing failed: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback

        logging.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PSCO model testing with feature importance analysis"
    )
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--processor_path", type=str, help="Path to data processor")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["permutation", "shap", "gradients", "integrated_gradients"],
        default=["permutation", "shap"],
        help="Feature importance methods to use",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Size of test sample to use for feature importance analysis (default: 1000)",
    )

    args = parser.parse_args()

    main(args.model_path, args.processor_path, args.methods, args.sample_size)
