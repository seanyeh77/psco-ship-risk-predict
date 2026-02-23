"""
Feature importance analysis utilities for PSCO model
"""

import os
import sys
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List
import seaborn as sns
from sklearn.metrics import accuracy_score
import pandas as pd

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def permutation_importance(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    device: torch.device,
    n_repeats: int = 10,
) -> Dict[str, float]:
    """
    Calculate permutation importance for each feature

    Args:
        model: Trained PyTorch model
        X: Input features
        y: True labels
        feature_names: List of feature names
        device: Device to run computations on
        n_repeats: Number of permutation repetitions

    Returns:
        Dictionary mapping feature names to importance scores
    """
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)

    # Get baseline accuracy
    with torch.no_grad():
        baseline_pred = model(X_tensor).argmax(dim=1)
        baseline_acc = accuracy_score(y, baseline_pred.cpu().numpy())

    importances = {}

    for i, feature_name in enumerate(feature_names):
        feature_importances = []

        for _ in range(n_repeats):
            X_permuted = X.copy()
            # Permute feature values
            np.random.shuffle(X_permuted[:, i])
            X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)

            with torch.no_grad():
                permuted_pred = model(X_permuted_tensor).argmax(dim=1)
                permuted_acc = accuracy_score(y, permuted_pred.cpu().numpy())

            # Importance is the decrease in accuracy
            importance = baseline_acc - permuted_acc
            feature_importances.append(importance)

        importances[feature_name] = np.mean(feature_importances)
        logging.info(
            f"Permutation importance for {feature_name}: {importances[feature_name]:.4f}"
        )

    return importances


def input_gradient_importance(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    device: torch.device,
) -> Dict[str, float]:
    """
    Calculate importance based on input gradients

    Args:
        model: Trained PyTorch model
        X: Input features
        y: True labels
        feature_names: List of feature names
        device: Device to run computations on

    Returns:
        Dictionary mapping feature names to importance scores
    """
    model.eval()
    model.zero_grad()

    X_tensor = torch.FloatTensor(X).to(device)
    X_tensor.requires_grad_(True)
    y_tensor = torch.LongTensor(y).to(device)

    # Forward pass
    outputs = model(X_tensor)

    # Calculate loss for correct predictions
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(outputs, y_tensor)

    # Backward pass
    loss.backward()

    # Get gradients - handle None case
    if X_tensor.grad is not None:
        gradients = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
    else:
        gradients = np.zeros(X_tensor.shape[1])

    importances = {}
    for i, feature_name in enumerate(feature_names):
        importances[feature_name] = gradients[i]

    return importances


def integrated_gradients_importance(
    model: torch.nn.Module,
    X: np.ndarray,
    feature_names: List[str],
    device: torch.device,
    steps: int = 50,
) -> Dict[str, float]:
    """
    Calculate integrated gradients importance

    Args:
        model: Trained PyTorch model
        X: Input features (subset recommended for computational efficiency)
        feature_names: List of feature names
        device: Device to run computations on
        steps: Number of integration steps

    Returns:
        Dictionary mapping feature names to importance scores
    """
    model.eval()

    X_tensor = torch.FloatTensor(X).to(device)
    baseline = torch.zeros_like(X_tensor).to(device)

    # Generate path from baseline to input
    alphas = torch.linspace(0, 1, steps).to(device)

    integrated_grads = torch.zeros_like(X_tensor).to(device)

    for alpha in alphas:
        # Interpolate between baseline and input
        interpolated_input = baseline + alpha * (X_tensor - baseline)
        interpolated_input.requires_grad_(True)

        # Forward pass
        outputs = model(interpolated_input)

        # Take gradient w.r.t. interpolated input
        # We'll use the max class prediction
        pred_class = outputs.argmax(dim=1)
        selected_outputs = outputs.gather(1, pred_class.unsqueeze(1))

        model.zero_grad()
        selected_outputs.sum().backward()

        # Accumulate gradients - handle None case
        if interpolated_input.grad is not None:
            integrated_grads += interpolated_input.grad

    # Average over path and multiply by (input - baseline)
    integrated_grads = integrated_grads / steps * (X_tensor - baseline)

    # Average importance across samples
    feature_importance = integrated_grads.abs().mean(dim=0).cpu().numpy()

    importances = {}
    for i, feature_name in enumerate(feature_names):
        importances[feature_name] = feature_importance[i]

    return importances


def shap_importance(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    device: torch.device,
    background_size: int = 100,
    sample_size: int = 500,
) -> Dict[str, float]:
    """
    Calculate SHAP (SHapley Additive exPlanations) importance for each feature

    Args:
        model: Trained PyTorch model
        X: Input features
        y: True labels
        feature_names: List of feature names
        device: Device to run computations on
        background_size: Size of background dataset for SHAP
        sample_size: Size of sample to calculate SHAP values for

    Returns:
        Dictionary mapping feature names to importance scores
    """
    try:
        import shap
    except ImportError:
        logging.error(
            "SHAP library not installed. Please install with: pip install shap"
        )
        raise ImportError("SHAP library is required for SHAP importance analysis")

    model.eval()

    # Prepare data - use smaller samples for efficiency
    sample_indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
    X_sample = X[sample_indices]

    background_indices = np.random.choice(
        len(X), min(background_size, len(X)), replace=False
    )
    X_background = X[background_indices]

    # Create a wrapper function for the model
    def model_wrapper(x):
        """Wrapper function that converts numpy to tensor and returns numpy"""
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(device)
            outputs = model(x_tensor)
            # Return probabilities for all classes
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()

    # Create SHAP explainer
    logging.info(
        f"Creating SHAP explainer with {len(X_background)} background samples..."
    )
    explainer = shap.Explainer(model_wrapper, X_background)

    # Calculate SHAP values
    logging.info(f"Calculating SHAP values for {len(X_sample)} samples...")
    shap_values = explainer(X_sample)

    # Handle multi-class case - use absolute values across all classes
    if len(shap_values.shape) > 2:
        # For multi-class, take mean absolute SHAP values across classes and samples
        feature_importance = np.abs(shap_values.values).mean(axis=(0, 2))
    else:
        # For binary classification, take mean absolute SHAP values across samples
        feature_importance = np.abs(shap_values.values).mean(axis=0)

    importances = {}
    for i, feature_name in enumerate(feature_names):
        importances[feature_name] = feature_importance[i]

    logging.info("SHAP importance calculation completed")
    return importances


def plot_feature_importance(
    importance_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    Plot feature importance comparison across different methods

    Args:
        importance_dict: Dictionary of {method_name: {feature_name: importance_score}}
        save_path: Path to save the plot
        figsize: Figure size multiplier (height will be method_count * figsize[1])
    """
    methods = list(importance_dict.keys())
    features = list(importance_dict[methods[0]].keys())

    fig, axes = plt.subplots(
        len(methods), 1, figsize=(figsize[0], figsize[1] * len(methods))
    )
    if len(methods) == 1:
        axes = [axes]

    for i, method in enumerate(methods):
        importance_values = [importance_dict[method][feature] for feature in features]

        # Sort by importance
        sorted_indices = np.argsort(importance_values)[::-1]
        sorted_features = [features[j] for j in sorted_indices]
        sorted_values = [importance_values[j] for j in sorted_indices]

        # Create horizontal bar plot
        y_pos = np.arange(len(sorted_features))
        bars = axes[i].barh(y_pos, sorted_values, color="skyblue", alpha=0.8)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(sorted_features)
        axes[i].set_xlabel("Importance Score")
        axes[i].set_title(f"Feature Importance - {method}")
        axes[i].grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, sorted_values)):
            axes[i].text(
                value + max(sorted_values) * 0.01,
                j,
                f"{value:.4f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Feature importance plot saved to: {save_path}")

    # plt.show()


def plot_feature_importance_heatmap(
    importance_dict: Dict[str, Dict[str, float]], save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance as a heatmap

    Args:
        importance_dict: Dictionary of {method_name: {feature_name: importance_score}}
        save_path: Path to save the plot
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(importance_dict).fillna(0)

    plt.figure(figsize=(10, len(df) * 0.4))
    sns.heatmap(
        df, annot=True, fmt=".4f", cmap="Blues", cbar_kws={"label": "Importance Score"}
    )
    plt.title("Feature Importance Heatmap")
    plt.xlabel("Methods")
    plt.ylabel("Features")
    plt.tight_layout()

    if save_path:
        heatmap_path = save_path.replace(".png", "_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        logging.info(f"Feature importance heatmap saved to: {heatmap_path}")

    # plt.show()


def save_feature_importance_csv(
    importance_dict: Dict[str, Dict[str, float]], save_path: str
) -> None:
    """
    Save feature importance results to CSV

    Args:
        importance_dict: Dictionary of {method_name: {feature_name: importance_score}}
        save_path: Path to save the CSV file
    """
    df = pd.DataFrame(importance_dict).fillna(0)
    df = df.round(6)  # Round to 6 decimal places

    # Add a rank column for each method
    for method in importance_dict.keys():
        rank_col = f"{method}_Rank"
        df[rank_col] = df[method].rank(ascending=False, method="dense").astype(int)

    df.to_csv(save_path)
    logging.info(f"Feature importance results saved to: {save_path}")


def analyze_feature_importance(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    device: torch.device,
    save_dir: str,
    timestamp: str,
    methods: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive feature importance analysis

    Args:
        model: Trained PyTorch model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        device: Device to run computations on
        save_dir: Directory to save results
        timestamp: Timestamp for file naming
        methods: List of methods to use ['permutation', 'shap', 'gradients', 'integrated_gradients']

    Returns:
        Dictionary mapping method names to feature importance dictionaries
    """
    if methods is None:
        methods = ["permutation", "shap"]

    logging.info("Analyzing feature importance...")

    importance_results = {}

    # 1. Permutation Importance
    if "permutation" in methods:
        logging.info("Computing permutation importance...")
        perm_importance = permutation_importance(
            model, X_test, y_test, feature_names, device
        )
        importance_results["Permutation"] = perm_importance

    # 2. SHAP Importance
    if "shap" in methods:
        logging.info("Computing SHAP importance...")
        shap_importance_result = shap_importance(
            model, X_test, y_test, feature_names, device
        )
        importance_results["SHAP"] = shap_importance_result

    # 3. Input Gradient Importance
    if "gradients" in methods:
        logging.info("Computing input gradient importance...")
        grad_importance = input_gradient_importance(
            model, X_test, y_test, feature_names, device
        )
        importance_results["Input Gradients"] = grad_importance

    # 4. Integrated Gradients Importance
    if "integrated_gradients" in methods:
        logging.info("Computing integrated gradients importance...")
        # Use a subset for computational efficiency
        subset_size = min(100, len(X_test))
        subset_indices = np.random.choice(len(X_test), subset_size, replace=False)
        X_subset = X_test[subset_indices]

        ig_importance = integrated_gradients_importance(
            model, X_subset, feature_names, device
        )
        importance_results["Integrated Gradients"] = ig_importance

    # Generate visualizations
    plot_path = os.path.join(save_dir, f"feature_importance_{timestamp}.png")
    plot_feature_importance(importance_results, plot_path)

    # Generate heatmap
    plot_feature_importance_heatmap(importance_results, plot_path)

    # Save to CSV
    csv_path = os.path.join(save_dir, f"feature_importance_{timestamp}.csv")
    save_feature_importance_csv(importance_results, csv_path)

    # Log top features for each method
    for method, importances in importance_results.items():
        logging.info(f"\nTop 5 features by {method}:")
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:5]):
            logging.info(f"  {i + 1}. {feature}: {importance:.4f}")

    return importance_results


def compare_feature_importance_methods(
    importance_results: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Compare feature rankings across different methods

    Args:
        importance_results: Dictionary mapping method names to feature importance dictionaries

    Returns:
        DataFrame with comparison statistics
    """
    df = pd.DataFrame(importance_results).fillna(0)

    # Add rank columns
    comparison_stats = []

    for feature in df.index:
        feature_stats = {"Feature": feature}

        # Get importance scores
        for method in df.columns:
            feature_stats[f"{method}_Score"] = df.loc[feature, method]
            feature_stats[f"{method}_Rank"] = (
                df[method].rank(ascending=False, method="dense").loc[feature]
            )

        # Calculate rank consistency (standard deviation of ranks)
        ranks = [feature_stats[f"{method}_Rank"] for method in df.columns]
        feature_stats["Rank_StdDev"] = np.std(ranks)
        feature_stats["Rank_Mean"] = np.mean(ranks)

        comparison_stats.append(feature_stats)

    comparison_df = pd.DataFrame(comparison_stats)
    comparison_df = comparison_df.sort_values("Rank_Mean")

    return comparison_df


if __name__ == "__main__":
    # Example usageÍ
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone feature importance analysis"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--processor_path", type=str, required=True, help="Path to data processor"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["permutation", "shap", "gradients", "integrated_gradients"],
        default=["permutation", "shap"],
        help="Methods to use for feature importance analysis",
    )
    parser.add_argument(
        "--output_dir", type=str, default="plots", help="Output directory"
    )

    args = parser.parse_args()

    # This would need to be implemented to load model and data
    print("Use this module by importing it in your main testing script.")
