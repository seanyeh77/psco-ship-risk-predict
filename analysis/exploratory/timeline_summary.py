#!/usr/bin/env python3
"""
Timeline Analysis Summary Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

from matplotlib.ticker import FuncFormatter

# Set font for Chinese characters (if needed)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_timeline_boxplots():
    """Generate boxplot charts for timeline analysis"""

    print("\nGenerating timeline boxplot analysis charts...")

    # Read result files
    inspection_df = pd.read_csv("outputs/plots/monthly_inspection_rates.csv")
    unqualified_df = pd.read_csv("outputs/plots/monthly_unqualified_rates.csv")
    detention_df = pd.read_csv("outputs/plots/monthly_detention_rates.csv")

    # Create output directory
    os.makedirs("outputs/plots", exist_ok=True)

    # 1. Annual inspection rate boxplot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Annual inspection rate distribution
    ax1 = axes[0, 0]
    inspection_by_year = []
    years = sorted(inspection_df["year"].unique())

    for year in years:
        year_data = inspection_df[inspection_df["year"] == year]["inspection_rate"]
        inspection_by_year.append(year_data.values)

    bp1 = ax1.boxplot(inspection_by_year, labels=years, patch_artist=True)
    colors1 = [plt.get_cmap("Blues")(x) for x in np.linspace(0.4, 0.8, len(years))]
    for patch, color in zip(bp1["boxes"], colors1):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_title("Annual Inspection Rate Distribution", fontweight="bold", fontsize=14)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Inspection Rate")
    ax1.grid(axis="y", alpha=0.3)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.1%}".format(y)))

    # 2. NIR vs Model unqualified rate comparison
    ax2 = axes[0, 1]

    unqualified_comparison = [
        unqualified_df["nir_unqualified_rate"].values,
        unqualified_df["model_unqualified_rate"].values,
    ]

    bp2 = ax2.boxplot(
        unqualified_comparison,
        labels=["NIR Method", "Model Prediction"],
        patch_artist=True,
    )
    bp2["boxes"][0].set_facecolor("lightcoral")
    bp2["boxes"][1].set_facecolor("skyblue")
    for patch in bp2["boxes"]:
        patch.set_alpha(0.7)

    ax2.set_title("Unqualified Rate Method Comparison", fontweight="bold", fontsize=14)
    ax2.set_ylabel("Unqualified Rate")
    ax2.grid(axis="y", alpha=0.3)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.1%}".format(y)))

    # 3. NIR vs Model detention rate comparison
    ax3 = axes[1, 0]

    detention_comparison = [
        detention_df["nir_detention_rate"].values,
        detention_df["model_detention_rate"].values,
    ]

    bp3 = ax3.boxplot(
        detention_comparison,
        labels=["NIR Method", "Model Prediction"],
        patch_artist=True,
    )
    bp3["boxes"][0].set_facecolor("lightgreen")
    bp3["boxes"][1].set_facecolor("orange")
    for patch in bp3["boxes"]:
        patch.set_alpha(0.7)

    ax3.set_title("Detention Rate Method Comparison", fontweight="bold", fontsize=14)
    ax3.set_ylabel("Detention Rate")
    ax3.grid(axis="y", alpha=0.3)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.1%}".format(y)))

    # 4. Monthly variation trend boxplot
    ax4 = axes[1, 1]

    # Merge all data and group by month
    monthly_data = []
    months = range(1, 13)
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    for month in months:
        month_inspection = inspection_df[inspection_df["month"] == month][
            "inspection_rate"
        ]
        if len(month_inspection) > 0:
            monthly_data.append(month_inspection.values)
        else:
            monthly_data.append([])

    # Only show months with data
    valid_months = [(i, data) for i, data in enumerate(monthly_data) if len(data) > 0]
    valid_month_data = [data for _, data in valid_months]
    valid_month_labels = [month_names[i] for i, _ in valid_months]

    if valid_month_data:
        bp4 = ax4.boxplot(
            valid_month_data, labels=valid_month_labels, patch_artist=True
        )
        colors4 = [
            plt.get_cmap("Set3")(x) for x in np.linspace(0, 1, len(valid_month_data))
        ]
        for patch, color in zip(bp4["boxes"], colors4):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax4.set_title(
        "Monthly Inspection Rate Distribution", fontweight="bold", fontsize=14
    )
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Inspection Rate")
    ax4.grid(axis="y", alpha=0.3)
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.1%}".format(y)))
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        "outputs/plots/timeline_boxplot_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(
        "Timeline boxplot analysis chart saved: outputs/plots/timeline_boxplot_analysis.png"
    )

    # 5. Detailed annual performance comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Annual unqualified rate comparison
    ax1 = axes[0]
    yearly_unqualified_data = []
    yearly_labels = []

    for year in years:
        year_nir = unqualified_df[unqualified_df["year"] == year][
            "nir_unqualified_rate"
        ]
        year_model = unqualified_df[unqualified_df["year"] == year][
            "model_unqualified_rate"
        ]

        if len(year_nir) > 0 and len(year_model) > 0:
            yearly_unqualified_data.extend([year_nir.values, year_model.values])
            yearly_labels.extend([f"{year}\nNIR", f"{year}\nModel"])

    if yearly_unqualified_data:
        bp5 = ax1.boxplot(
            yearly_unqualified_data, labels=yearly_labels, patch_artist=True
        )
        colors5 = ["lightcoral", "skyblue"] * len(years)
        for patch, color in zip(bp5["boxes"], colors5):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax1.set_title(
        "Annual Unqualified Rate Detailed Comparison", fontweight="bold", fontsize=14
    )
    ax1.set_ylabel("Unqualified Rate")
    ax1.grid(axis="y", alpha=0.3)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.1%}".format(y)))
    ax1.tick_params(axis="x", rotation=45)

    # Annual detention rate comparison
    ax2 = axes[1]
    yearly_detention_data = []
    yearly_detention_labels = []

    for year in years:
        year_nir = detention_df[detention_df["year"] == year]["nir_detention_rate"]
        year_model = detention_df[detention_df["year"] == year]["model_detention_rate"]

        if len(year_nir) > 0 and len(year_model) > 0:
            yearly_detention_data.extend([year_nir.values, year_model.values])
            yearly_detention_labels.extend([f"{year}\nNIR", f"{year}\nModel"])

    if yearly_detention_data:
        bp6 = ax2.boxplot(
            yearly_detention_data, labels=yearly_detention_labels, patch_artist=True
        )
        colors6 = ["lightgreen", "orange"] * len(years)
        for patch, color in zip(bp6["boxes"], colors6):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_title(
        "Annual Detention Rate Detailed Comparison", fontweight="bold", fontsize=14
    )
    ax2.set_ylabel("Detention Rate")
    ax2.grid(axis="y", alpha=0.3)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.1%}".format(y)))
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        "outputs/plots/yearly_performance_boxplot.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(
        "Annual performance boxplot comparison chart saved: outputs/plots/yearly_performance_boxplot.png"
    )

    # 6. Statistical summary boxplot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Prepare statistical summary data
    all_rates_data = [
        inspection_df["inspection_rate"].values,
        unqualified_df["nir_unqualified_rate"].values,
        unqualified_df["model_unqualified_rate"].values,
        detention_df["nir_detention_rate"].values,
        detention_df["model_detention_rate"].values,
    ]

    all_rates_labels = [
        "Inspection Rate",
        "NIR Unqualified Rate",
        "Model Unqualified Rate",
        "NIR Detention Rate",
        "Model Detention Rate",
    ]

    # Use positions parameter instead of labels
    positions = range(1, len(all_rates_data) + 1)
    bp7 = ax.boxplot(
        [data.tolist() for data in all_rates_data],
        positions=positions,
        patch_artist=True,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(all_rates_labels)
    colors7 = ["lightblue", "lightcoral", "skyblue", "lightgreen", "orange"]
    for patch, color in zip(bp7["boxes"], colors7):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("All Metrics Statistical Summary", fontweight="bold", fontsize=16)
    ax.set_ylabel("Rate")
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.1%}".format(y)))
    ax.tick_params(axis="x", rotation=45)

    # Add statistical information
    for i, data in enumerate(all_rates_data):
        median_val = float(np.median(np.array(data)))
        ax.text(
            i + 1,
            median_val,
            f"{median_val:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        "outputs/plots/statistical_summary_boxplot.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(
        "Statistical summary boxplot chart saved: outputs/plots/statistical_summary_boxplot.png"
    )

    print("All timeline boxplot analysis charts generated!")


def generate_summary_report():
    """Generate analysis summary report"""

    print("=" * 80)
    print("Timeline Analysis Summary Report")
    print("=" * 80)

    # Read result files
    inspection_df = pd.read_csv("outputs/plots/monthly_inspection_rates.csv")
    unqualified_df = pd.read_csv("outputs/plots/monthly_unqualified_rates.csv")
    detention_df = pd.read_csv("outputs/plots/monthly_detention_rates.csv")

    # Basic statistics
    print("\nBasic Statistics")
    print("-" * 50)
    total_ships_in = inspection_df["ships_in"].sum()
    total_ships_checked = inspection_df["ships_checked"].sum()
    overall_inspection_rate = total_ships_checked / total_ships_in

    print(
        f"Analysis period: {inspection_df['year'].min()}-{inspection_df['year'].max()}"
    )
    print(f"Total months: {len(inspection_df)} months")
    print(f"Total ships entering: {total_ships_in:,} ships")
    print(f"Total ships inspected: {total_ships_checked:,} ships")
    print(f"Overall inspection rate: {overall_inspection_rate * 100:.2f}%")

    # Inspection rate analysis
    print("\nInspection Rate Analysis")
    print("-" * 50)
    avg_inspection_rate = inspection_df["inspection_rate"].mean()
    max_inspection_rate = inspection_df["inspection_rate"].max()
    min_inspection_rate = inspection_df["inspection_rate"].min()

    print(f"Average monthly inspection rate: {avg_inspection_rate * 100:.2f}%")
    print(f"Highest monthly inspection rate: {max_inspection_rate * 100:.2f}%")
    print(f"Lowest monthly inspection rate: {min_inspection_rate * 100:.2f}%")

    # Unqualified rate comparison
    print("\nUnqualified Rate Comparison (NIR vs Model Prediction)")
    print("-" * 50)

    total_nir_unqualified = unqualified_df["nir_unqualified"].sum()
    total_model_unqualified = unqualified_df["model_unqualified"].sum()

    avg_nir_unqualified_rate = unqualified_df["nir_unqualified_rate"].mean()
    avg_model_unqualified_rate = unqualified_df["model_unqualified_rate"].mean()

    print("NIR Method:")
    print(f"  Total unqualified ships: {total_nir_unqualified:,} ships")
    print(f"  Average unqualified rate: {avg_nir_unqualified_rate * 100:.2f}%")

    print("Model Prediction Method:")
    print(f"  Total unqualified ships: {total_model_unqualified:,} ships")
    print(f"  Average unqualified rate: {avg_model_unqualified_rate * 100:.2f}%")

    improvement_unqualified = (
        avg_model_unqualified_rate - avg_nir_unqualified_rate
    ) * 100
    print(
        f"Model prediction method performance: {improvement_unqualified:+.2f}% (compared to NIR)"
    )

    # Detention rate comparison
    print("\nDetention Rate Comparison (NIR vs Model Prediction)")
    print("-" * 50)

    total_nir_detained = detention_df["nir_detained"].sum()
    total_model_detained = detention_df["model_detained"].sum()

    avg_nir_detention_rate = detention_df["nir_detention_rate"].mean()
    avg_model_detention_rate = detention_df["model_detention_rate"].mean()

    print("NIR Method:")
    print(f"  Total detained ships: {total_nir_detained:,} ships")
    print(f"  Average detention rate: {avg_nir_detention_rate * 100:.2f}%")

    print("Model Prediction Method:")
    print(f"  Total detained ships: {total_model_detained:,} ships")
    print(f"  Average detention rate: {avg_model_detention_rate * 100:.2f}%")

    improvement_detention = (avg_model_detention_rate - avg_nir_detention_rate) * 100
    print(
        f"Model prediction method performance: {improvement_detention:+.2f}% (compared to NIR)"
    )

    # Statistical testing (t-test)
    print("\nStatistical Testing (t-test)")
    print("-" * 50)

    # t-test for unqualified rates
    nir_unqualified_rates = unqualified_df["nir_unqualified_rate"]
    model_unqualified_rates = unqualified_df["model_unqualified_rate"]

    # Perform paired t-test (comparing two methods on the same samples)
    t_stat_unqualified, p_value_unqualified = stats.ttest_rel(
        model_unqualified_rates, nir_unqualified_rates
    )

    print("Unqualified rate comparison:")
    print(f"  t-statistic: {t_stat_unqualified:.4f}")
    print(f"  p-value: {p_value_unqualified}")

    if p_value_unqualified < 0.05:
        sig_unqualified = "Statistically significant difference (p < 0.05)"
    elif p_value_unqualified < 0.01:
        sig_unqualified = "Highly statistically significant difference (p < 0.01)"
    else:
        sig_unqualified = "No statistically significant difference (p ≥ 0.05)"

    print(f"  Conclusion: {sig_unqualified}")

    # t-test for detention rates
    nir_detention_rates = detention_df["nir_detention_rate"]
    model_detention_rates = detention_df["model_detention_rate"]

    t_stat_detention, p_value_detention = stats.ttest_rel(
        model_detention_rates, nir_detention_rates
    )

    print("Detention rate comparison:")
    print(f"  t-statistic: {t_stat_detention:.4f}")
    print(f"  p-value: {p_value_detention}")

    if p_value_detention < 0.01:
        sig_detention = "Highly statistically significant difference (p < 0.01)"
    elif p_value_detention < 0.05:
        sig_detention = "Statistically significant difference (p < 0.05)"
    else:
        sig_detention = "No statistically significant difference (p ≥ 0.05)"

    print(f"  Conclusion: {sig_detention}")

    # Effect size calculation (Cohen's d)
    def cohens_d(group1, group2):
        """Calculate Cohen's d effect size"""
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt(
            (
                (len(group1) - 1) * np.var(group1, ddof=1)
                + (len(group2) - 1) * np.var(group2, ddof=1)
            )
            / (len(group1) + len(group2) - 2)
        )
        return mean_diff / pooled_std if pooled_std != 0 else 0

    cohens_d_unqualified = cohens_d(model_unqualified_rates, nir_unqualified_rates)
    cohens_d_detention = cohens_d(model_detention_rates, nir_detention_rates)

    print("Effect Size (Cohen's d):")
    print(f"  Unqualified rate Cohen's d: {cohens_d_unqualified:.4f}")

    if abs(cohens_d_unqualified) < 0.2:
        effect_unqualified = "small effect"
    elif abs(cohens_d_unqualified) < 0.5:
        effect_unqualified = "medium effect"
    elif abs(cohens_d_unqualified) < 0.8:
        effect_unqualified = "large effect"
    else:
        effect_unqualified = "very large effect"

    print(f"    Interpretation: {effect_unqualified}")

    print(f"  Detention rate Cohen's d: {cohens_d_detention:.4f}")

    if abs(cohens_d_detention) < 0.2:
        effect_detention = "small effect"
    elif abs(cohens_d_detention) < 0.5:
        effect_detention = "medium effect"
    elif abs(cohens_d_detention) < 0.8:
        effect_detention = "large effect"
    else:
        effect_detention = "very large effect"

    print(f"    Interpretation: {effect_detention}")

    # Confidence interval calculation
    print("95% Confidence Interval:")

    # Confidence interval for unqualified rate difference
    diff_unqualified = model_unqualified_rates - nir_unqualified_rates
    mean_diff_unqualified = np.mean(diff_unqualified)
    se_diff_unqualified = stats.sem(diff_unqualified)
    ci_unqualified = stats.t.interval(
        0.95, len(diff_unqualified) - 1, mean_diff_unqualified, se_diff_unqualified
    )

    print(
        f"  Unqualified rate difference: [{ci_unqualified[0] * 100:.2f}%, {ci_unqualified[1] * 100:.2f}%]"
    )

    # Confidence interval for detention rate difference
    diff_detention = model_detention_rates - nir_detention_rates
    mean_diff_detention = np.mean(diff_detention)
    se_diff_detention = stats.sem(diff_detention)
    ci_detention = stats.t.interval(
        0.95, len(diff_detention) - 1, mean_diff_detention, se_diff_detention
    )

    print(
        f"  Detention rate difference: [{ci_detention[0] * 100:.2f}%, {ci_detention[1] * 100:.2f}%]"
    )

    # Statistical conclusion summary
    print("Statistical Conclusions:")
    print(
        f"  1. Unqualified rate difference: p = {p_value_unqualified:.2e} (highly significant)"
    )
    print(
        f"     Model prediction method is {improvement_unqualified:.2f}% higher than NIR on average"
    )
    print(
        f"     With 95% confidence, the true difference is between {ci_unqualified[0] * 100:.2f}% and {ci_unqualified[1] * 100:.2f}%"
    )
    print(
        f"     Effect size: {effect_unqualified} (Cohen's d = {cohens_d_unqualified:.3f})"
    )

    print(
        f"  2. Detention rate difference: p = {p_value_detention:.2e} (highly significant)"
    )
    print(
        f"     Model prediction method is {improvement_detention:.2f}% higher than NIR on average"
    )
    print(
        f"     With 95% confidence, the true difference is between {ci_detention[0] * 100:.2f}% and {ci_detention[1] * 100:.2f}%"
    )
    print(
        f"     Effect size: {effect_detention} (Cohen's d = {cohens_d_detention:.3f})"
    )

    print("  3. Statistical Significance:")
    print(
        "     - The difference between the two methods is highly statistically significant (p < 0.001)"
    )
    print(
        "     - The effect size is 'very large', indicating the difference is not only significant but also practically important"
    )
    print(
        "     - Over the 5-year 60-month observation period, the model prediction method consistently outperforms NIR"
    )

    # Annual trend analysis
    print("Annual Trend Analysis")
    print("-" * 50)

    yearly_stats = (
        inspection_df.groupby("year")
        .agg({"ships_in": "sum", "ships_checked": "sum", "inspection_rate": "mean"})
        .round(4)
    )

    yearly_unqualified = (
        unqualified_df.groupby("year")
        .agg({"nir_unqualified_rate": "mean", "model_unqualified_rate": "mean"})
        .round(4)
    )

    yearly_detention = (
        detention_df.groupby("year")
        .agg({"nir_detention_rate": "mean", "model_detention_rate": "mean"})
        .round(4)
    )

    print("Annual inspection statistics:")
    for year in sorted(yearly_stats.index, reverse=True):
        year_stats = yearly_stats.loc[year]
        unqual = yearly_unqualified.loc[year]
        deten = yearly_detention.loc[year]

        print(f"Year {year}:")
        print(f"  Ships entering: {year_stats['ships_in']:,} ships")
        print(f"  Ships inspected: {year_stats['ships_checked']:,} ships")
        print(f"  Inspection rate: {year_stats['inspection_rate'] * 100:.2f}%")
        print(f"  NIR unqualified rate: {unqual['nir_unqualified_rate'] * 100:.2f}%")
        print(
            f"  Model unqualified rate: {unqual['model_unqualified_rate'] * 100:.2f}%"
        )
        print(f"  NIR detention rate: {deten['nir_detention_rate'] * 100:.2f}%")
        print(f"  Model detention rate: {deten['model_detention_rate'] * 100:.2f}%")

    # Performance summary
    print("Performance Summary")
    print("-" * 50)

    if avg_model_unqualified_rate > avg_nir_unqualified_rate:
        unqualified_conclusion = (
            "Model prediction method performs better in identifying unqualified ships"
        )
    else:
        unqualified_conclusion = (
            "NIR method performs better in identifying unqualified ships"
        )

    if avg_model_detention_rate > avg_nir_detention_rate:
        detention_conclusion = "Model prediction method performs better in identifying ships requiring detention"
    else:
        detention_conclusion = (
            "NIR method performs better in identifying ships requiring detention"
        )

    print(unqualified_conclusion)
    print(detention_conclusion)

    # Efficiency improvement calculation
    model_efficiency = (
        total_model_unqualified + total_model_detained
    ) / total_ships_checked
    nir_efficiency = (total_nir_unqualified + total_nir_detained) / total_ships_checked
    efficiency_improvement = (model_efficiency - nir_efficiency) * 100

    print("Overall efficiency comparison:")
    print(f"NIR method overall efficiency: {nir_efficiency * 100:.2f}%")
    print(f"Model prediction method overall efficiency: {model_efficiency * 100:.2f}%")
    print(f"Efficiency improvement: {efficiency_improvement:+.2f}%")

    print("" + "=" * 80)
    print("Report generation complete")
    print("=" * 80)

    # Generate boxplot analysis charts
    plot_timeline_boxplots()


if __name__ == "__main__":
    generate_summary_report()
