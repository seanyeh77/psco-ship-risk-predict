"""
Comprehensive analysis and comparison between PSCO model and NIR system predictions
Loads prediction results from outputs/predictions/ and performs detailed comparison analysis
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
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


def load_latest_predictions(data_dir: str) -> pd.DataFrame:
    """
    Load the latest merged prediction results
    """
    merged_files = glob.glob(os.path.join(data_dir, "merged_predictions_*.csv"))
    if not merged_files:
        raise FileNotFoundError(
            "No merged prediction files found. Run generate_predictions.py first."
        )

    # Get the latest file
    latest_file = max(merged_files, key=os.path.getctime)
    print(f"Loading predictions from: {os.path.basename(latest_file)}")

    df = pd.read_csv(latest_file)
    print(f"Loaded {len(df)} samples for comparison")

    return df


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


def plot_risk_distribution_comparison(df, output_dir):
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

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "risk_distribution_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Risk distribution comparison plot saved")


def plot_confusion_matrices(df, output_dir):
    """Plot confusion matrix comparison"""
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
        os.path.join(output_dir, "confusion_matrices_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Confusion matrices comparison plot saved")


def plot_performance_metrics(df, output_dir):
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
        os.path.join(output_dir, "performance_metrics_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Performance metrics comparison plot saved")

    return psco_metrics, nir_metrics


def plot_prediction_agreement(df, output_dir):
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
        os.path.join(output_dir, "risk_agreement_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Prediction agreement heatmap saved")

    return agreement_table


def generate_summary_report(psco_metrics, nir_metrics, df, output_dir):
    """Generate detailed comparison report and save to file"""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(
        "                    PSCO vs NIR Model Comparison Analysis Report"
    )
    report_lines.append("=" * 80)

    report_lines.append("\nData Overview:")
    report_lines.append(f"  Total samples: {len(df)}")
    report_lines.append("  True risk distribution:")
    for risk_level in sorted(df["true_label"].unique()):
        count = (df["true_label"] == risk_level).sum()
        pct = count / len(df) * 100
        report_lines.append(f"    Risk level {risk_level}: {count} ({pct:.1f}%)")

    report_lines.append("\nPSCO Model Prediction Distribution:")
    risk_labels = {0: "Low Risk", 1: "Standard Risk", 2: "High Risk"}
    for risk_code, risk_name in risk_labels.items():
        count = (df["psco_prediction"] == risk_code).sum()
        pct = count / len(df) * 100
        report_lines.append(f"  {risk_name}: {count} ({pct:.1f}%)")

    report_lines.append("\nNIR Model Prediction Distribution:")
    for category in ["LRS", "SRS", "HRS"]:
        count = (df["risk_category"] == category).sum()
        pct = count / len(df) * 100
        report_lines.append(f"  {category}: {count} ({pct:.1f}%)")

    report_lines.append("\nPerformance Metrics Comparison:")
    report_lines.append(
        f"{'Metric':<15} {'PSCO Model':<15} {'NIR Model':<15} {'PSCO Advantage':<15}"
    )
    report_lines.append("-" * 65)

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
        report_lines.append(
            f"{name:<15} {psco_val:<15.3f} {nir_val:<15.3f} {diff:+.3f}"
        )

    # Calculate prediction agreement
    agreement = (df["psco_prediction"] == df["nir_prediction"]).mean()
    report_lines.append(f"\nPrediction Agreement: {agreement:.2%}")

    # Analyze high-risk prediction capability
    high_risk_psco = (df["psco_prediction"] == 2).sum()
    high_risk_nir = (df["nir_prediction"] == 2).sum()
    actual_high_risk = (df["true_label"] == 2).sum()

    report_lines.append("\nHigh Risk Prediction Analysis:")
    report_lines.append(f"  Actual high-risk vessels: {actual_high_risk}")
    report_lines.append(f"  PSCO predicted high-risk: {high_risk_psco}")
    report_lines.append(f"  NIR predicted high-risk: {high_risk_nir}")

    # Conclusion
    report_lines.append("\nConclusion:")
    if psco_metrics["f1_score"] > nir_metrics["f1_score"]:
        improvement = psco_metrics["f1_score"] - nir_metrics["f1_score"]
        report_lines.append(
            "  PSCO model significantly outperforms NIR model in overall performance"
        )
        report_lines.append(
            f"  F1-Score improvement: +{improvement:.3f} ({improvement / nir_metrics['f1_score'] * 100:.1f}%)"
        )
    else:
        report_lines.append(
            "  NIR model may have advantages in some metrics, requires further analysis"
        )

    report_lines.append("\nGenerated Analysis Files:")
    report_lines.append("  - Risk distribution comparison")
    report_lines.append("  - Confusion matrices comparison")
    report_lines.append("  - Performance metrics comparison")
    report_lines.append("  - Prediction agreement heatmap")

    # Print to console
    for line in report_lines:
        print(line)

    # Save to file
    report_path = os.path.join(output_dir, "comparison_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nDetailed report saved to: {report_path}")


def main():
    """Main function"""
    try:
        print("Starting comparison analysis...")

        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_dir = os.path.join(project_root, "outputs", "predictions")
        output_dir = os.path.join(project_root, "outputs", "plots")
        os.makedirs(output_dir, exist_ok=True)

        # Load latest prediction results
        df = load_latest_predictions(data_dir)

        # Generate all analysis plots
        print("\nGenerating analysis charts...")
        plot_risk_distribution_comparison(df, output_dir)
        plot_confusion_matrices(df, output_dir)
        psco_metrics, nir_metrics = plot_performance_metrics(df, output_dir)
        plot_prediction_agreement(df, output_dir)

        # Generate detailed report
        print("\nGenerating comparison report...")
        generate_summary_report(psco_metrics, nir_metrics, df, output_dir)

        print("\nComparison analysis completed!")
        print(f"All analysis files saved to: {output_dir}")

    except Exception as e:
        print(f"Error occurred during analysis: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
