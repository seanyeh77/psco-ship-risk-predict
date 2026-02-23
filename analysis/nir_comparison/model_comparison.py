"""
Comprehensive model comparison script between PSCO model and NIR system
Analyzes performance, risk distribution, and prediction consistency
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Optional
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import warnings

warnings.filterwarnings("ignore")

# Add src directory to path for psco package
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from psco.config import Config
from psco.data_processor import DataProcessor
from psco.model import create_model
from psco.trainer import load_model

# # Set font for better display
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
# plt.rcParams['axes.unicode_minus'] = False


def load_psco_model_and_predict(
    data_path: str,
    model_path: Optional[str] = None,
    processor_path: Optional[str] = None,
):
    """
    Load PSCO model and make predictions
    """
    print("Loading PSCO model...")

    # Use latest model file if not specified
    if model_path is None:
        models_dir = os.path.join(project_root, "models")
        model_files = [
            f
            for f in os.listdir(models_dir)
            if f.startswith("psco_model_") and f.endswith(".pth")
        ]
        if model_files:
            model_path = os.path.join(models_dir, sorted(model_files)[-1])
        else:
            raise FileNotFoundError("PSCO model file not found")

    if processor_path is None:
        models_dir = os.path.join(project_root, "models")
        processor_files = [
            f
            for f in os.listdir(models_dir)
            if f.startswith("data_processor_") and f.endswith(".pth")
        ]
        if processor_files:
            processor_path = os.path.join(models_dir, sorted(processor_files)[-1])
        else:
            raise FileNotFoundError("Data processor file not found")

    print(f"Using model: {os.path.basename(model_path)}")
    print(f"Using processor: {os.path.basename(processor_path)}")

    # Load configuration
    config = Config()

    # Load data processor
    processor_data = torch.load(processor_path, map_location="cpu", weights_only=False)
    data_processor = DataProcessor(config)
    data_processor.scaler = processor_data["scaler"]
    data_processor.class_weights = processor_data["class_weights"]

    # Load data
    df = pd.read_csv(data_path)

    # Filter test data (inspect==0)
    test_data = df[df["inspect"] == 0].copy()
    print(f"Test data samples: {len(test_data)}")

    # Process data
    df_test = data_processor.preprocess_data(test_data, data_type="test")
    X_test, y_test = data_processor.transform_test_data(df_test)
    test_ids = test_data["ID"].values

    # Load model
    device = torch.device("cpu")
    model = create_model(config=config)
    epoch, metrics = load_model(model, model_path)
    model.to(device)
    model.eval()

    # Make predictions
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "ID": test_ids,
            "psco_prediction": predictions.cpu().numpy(),
            "psco_prob_0": probabilities[:, 0].cpu().numpy(),
            "psco_prob_1": probabilities[:, 1].cpu().numpy(),
            "psco_prob_2": probabilities[:, 2].cpu().numpy(),
            "true_label": y_test,
        }
    )

    return results_df, test_data


def load_nir_predictions():
    """
    Load NIR prediction results
    """
    print("Loading NIR predictions...")

    # Import and run NIR prediction script directly
    try:
        # Import the NIR module
        import add_NIR_columns

        # Run the main function to generate predictions
        add_NIR_columns.main()
        print("NIR prediction completed successfully")

    except ImportError as e:
        print(f"Error importing NIR module: {e}")
        raise
    except Exception as e:
        print(f"Error running NIR predictions: {e}")
        raise

    # Load processed data
    processed_data_path = os.path.join(
        project_root, "data", "processed", "processed_data.csv"
    )
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(
            f"NIR processed data file not found: {processed_data_path}"
        )

    nir_df = pd.read_csv(processed_data_path)

    # Convert NIR risk categories to numerical values
    nir_risk_map = {"LRS": 0, "SRS": 1, "HRS": 2}
    nir_df["nir_prediction"] = nir_df["risk_category"].map(nir_risk_map)

    return nir_df[
        ["ID", "nir_prediction", "risk_category", "weighting_point", "bgw_list"]
    ]


def merge_predictions(psco_results, nir_results, test_data):
    """
    Merge PSCO and NIR prediction results
    """
    # Merge prediction results
    merged_df = psco_results.merge(nir_results, on="ID", how="inner")

    # Add original features
    test_features = test_data[
        ["ID", "detained", "qualified", "YOB", "GT", "NoInsp", "NoDef"]
    ].copy()
    merged_df = merged_df.merge(test_features, on="ID", how="inner")

    print(f"Merged samples: {len(merged_df)}")
    return merged_df


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def plot_risk_distribution_comparison(df):
    """Plot risk distribution comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # PSCO model risk distribution
    psco_risk_labels = {0: "Low Risk", 1: "Standard Risk", 2: "High Risk"}
    psco_counts = df["psco_prediction"].map(psco_risk_labels).value_counts()
    colors1 = ["lightgreen", "gold", "lightcoral"]
    ax1.pie(
        psco_counts.values,
        labels=psco_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors1,
    )
    ax1.set_title("PSCO Model Risk Distribution", fontsize=14, fontweight="bold")

    # NIR model risk distribution
    nir_counts = df["risk_category"].value_counts()
    colors2 = ["lightblue", "lightyellow", "lightpink"]
    ax2.pie(
        nir_counts.values,
        labels=nir_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors2,
    )
    ax2.set_title("NIR Model Risk Distribution", fontsize=14, fontweight="bold")

    os.makedirs("./outputs/plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        "./outputs/plots/risk_distribution_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_confusion_matrices(df):
    """Plot confusion matrix comparison"""
    # Use true labels as reference
    true_labels = df["true_label"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # PSCO model confusion matrix
    cm_psco = confusion_matrix(true_labels, df["psco_prediction"])
    sns.heatmap(cm_psco, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_title("PSCO Model Confusion Matrix", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")

    # NIR model confusion matrix
    cm_nir = confusion_matrix(true_labels, df["nir_prediction"])
    sns.heatmap(cm_nir, annot=True, fmt="d", cmap="Greens", ax=ax2)
    ax2.set_title("NIR Model Confusion Matrix", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(
        "./outputs/plots/confusion_matrices_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_performance_metrics(df):
    """Plot performance metrics comparison"""
    true_labels = df["true_label"].values

    # Calculate metrics
    psco_metrics = calculate_metrics(true_labels, df["psco_prediction"])
    nir_metrics = calculate_metrics(true_labels, df["nir_prediction"])

    # Prepare data
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    psco_values = [
        psco_metrics["accuracy"],
        psco_metrics["precision"],
        psco_metrics["recall"],
        psco_metrics["f1_score"],
    ]
    nir_values = [
        nir_metrics["accuracy"],
        nir_metrics["precision"],
        nir_metrics["recall"],
        nir_metrics["f1_score"],
    ]

    # Plot comparison
    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(
        x - width / 2,
        psco_values,
        width,
        label="PSCO Model",
        color="skyblue",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        nir_values,
        width,
        label="NIR Model",
        color="lightcoral",
        alpha=0.8,
    )

    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(
            bar1.get_x() + bar1.get_width() / 2.0,
            height1 + 0.01,
            f"{height1:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        ax.text(
            bar2.get_x() + bar2.get_width() / 2.0,
            height2 + 0.01,
            f"{height2:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_xlabel("Evaluation Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "./outputs/plots/performance_metrics_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return psco_metrics, nir_metrics


def plot_prediction_agreement(df):
    """Plot prediction agreement analysis"""
    # Create cross table
    risk_labels = {0: "Low Risk", 1: "Standard Risk", 2: "High Risk"}
    df_temp = df.copy()
    df_temp["psco_risk_label"] = df_temp["psco_prediction"].map(risk_labels)

    agreement_table = pd.crosstab(df_temp["risk_category"], df_temp["psco_risk_label"])

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        agreement_table,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar_kws={"label": "Sample Count"},
    )
    plt.title(
        "Prediction Agreement Analysis\n(NIR Model vs PSCO Model)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("PSCO Model Prediction", fontsize=12, fontweight="bold")
    plt.ylabel("NIR Model Prediction", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        "./outputs/plots/risk_agreement_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    return agreement_table


def generate_summary_report(psco_metrics, nir_metrics, df):
    """Generate detailed comparison report"""
    print("=" * 80)
    print("                    PSCO vs NIR Model Comparison Analysis Report")
    print("=" * 80)

    print("\nData Overview:")
    print(f"  Total samples: {len(df)}")
    print("  True risk distribution:")
    for risk_level in sorted(df["true_label"].unique()):
        count = (df["true_label"] == risk_level).sum()
        pct = count / len(df) * 100
        print(f"    Risk level {risk_level}: {count} ({pct:.1f}%)")

    print("\nPSCO Model Prediction Distribution:")
    risk_labels = {0: "Low Risk", 1: "Standard Risk", 2: "High Risk"}
    for risk_code, risk_name in risk_labels.items():
        count = (df["psco_prediction"] == risk_code).sum()
        pct = count / len(df) * 100
        print(f"  {risk_name}: {count} ({pct:.1f}%)")

    print("\nNIR Model Prediction Distribution:")
    for category in ["LRS", "SRS", "HRS"]:
        count = (df["risk_category"] == category).sum()
        pct = count / len(df) * 100
        print(f"  {category}: {count} ({pct:.1f}%)")

    print("\nPerformance Metrics Comparison:")
    print(f"{'Metric':<15} {'PSCO Model':<15} {'NIR Model':<15} {'PSCO Advantage':<15}")
    print("-" * 65)

    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    psco_values = [
        psco_metrics["accuracy"],
        psco_metrics["precision"],
        psco_metrics["recall"],
        psco_metrics["f1_score"],
    ]
    nir_values = [
        nir_metrics["accuracy"],
        nir_metrics["precision"],
        nir_metrics["recall"],
        nir_metrics["f1_score"],
    ]

    for name, psco_val, nir_val in zip(metrics_names, psco_values, nir_values):
        diff = psco_val - nir_val
        print(f"{name:<15} {psco_val:<15.3f} {nir_val:<15.3f} {diff:+.3f}")

    # Calculate prediction agreement
    agreement = (df["psco_prediction"] == df["nir_prediction"]).mean()
    print(f"\nPrediction Agreement: {agreement:.2%}")

    # Analyze high-risk prediction capability
    high_risk_psco = (df["psco_prediction"] == 2).sum()
    high_risk_nir = (df["nir_prediction"] == 2).sum()
    actual_high_risk = (df["true_label"] == 2).sum()

    print("\nHigh Risk Prediction Analysis:")
    print(f"  Actual high-risk vessels: {actual_high_risk}")
    print(f"  PSCO predicted high-risk: {high_risk_psco}")
    print(f"  NIR predicted high-risk: {high_risk_nir}")

    # Conclusion
    print("\nConclusion:")
    if psco_metrics["f1_score"] > nir_metrics["f1_score"]:
        improvement = psco_metrics["f1_score"] - nir_metrics["f1_score"]
        print("  PSCO model significantly outperforms NIR model in overall performance")
        print(
            f"  F1-Score improvement: +{improvement:.3f} ({improvement / nir_metrics['f1_score'] * 100:.1f}%)"
        )
    else:
        print(
            "  NIR model may have advantages in some metrics, requires further analysis"
        )

    print("\nAll analysis charts have been saved to outputs/plots/ directory")


def main():
    """Main function"""
    try:
        print("Starting model comparison analysis...")

        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))

        # Data path
        data_path = os.path.join(
            project_root, "data", "simulated", "simulated_data.csv"
        )

        # 1. Load PSCO model and predict
        psco_results, test_data = load_psco_model_and_predict(data_path)
        print(f"PSCO model prediction completed: {len(psco_results)} samples")

        # 2. Load NIR prediction results
        nir_results = load_nir_predictions()
        print(f"NIR prediction loading completed: {len(nir_results)} samples")

        # 3. Merge prediction results
        merged_df = merge_predictions(psco_results, nir_results, test_data)
        print(f"Data merging completed: {len(merged_df)} samples for comparison")

        # 4. Generate comparison charts
        print("\nGenerating analysis charts...")
        plot_risk_distribution_comparison(merged_df)
        plot_confusion_matrices(merged_df)
        psco_metrics, nir_metrics = plot_performance_metrics(merged_df)
        plot_prediction_agreement(merged_df)

        # 5. Generate detailed report
        print("\nGenerating comparison report...")
        generate_summary_report(psco_metrics, nir_metrics, merged_df)

        print("\nModel comparison analysis completed!")

    except Exception as e:
        print(f"Error occurred during analysis: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
