"""
Timeline Analysis Based on Historical Data
Create monthly analysis based on timeline.csv, comparing NIR method and model prediction method
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
from pathlib import Path

# Add src directory to path for psco package
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "analysis" / "nir_comparison"))
from psco.model import create_model
from psco.data_processor import DataProcessor
from psco.config import Config
from psco.trainer import load_model
from nir_comparison.add_NIR_columns import add_shipping_info, nir_predict 


class TimelineAnalyzer:
    def __init__(self):
        self.config = Config()
        self.model = None
        self.processor = None

    def load_model_and_processor(self):
        """Load model and data processor"""
        try:
            # Find the latest model files
            model_files = list(Path("models/").glob("psco_model_*.pth"))
            processor_files = list(Path("models/").glob("data_processor_*.pth"))

            if not model_files or not processor_files:
                raise FileNotFoundError("Model or processor files not found")

            # Use the latest files
            latest_model = max(model_files, key=os.path.getctime)
            latest_processor = max(processor_files, key=os.path.getctime)

            print(f"Loading model: {latest_model}")
            print(f"Loading processor: {latest_processor}")

            # Load processor
            processor_data = torch.load(
                latest_processor, map_location="cpu", weights_only=False
            )
            self.processor = DataProcessor(self.config)
            self.processor.scaler = processor_data["scaler"]
            self.processor.class_weights = processor_data["class_weights"]

            # Load model
            self.model = create_model(config=self.config)
            epoch, metrics = load_model(self.model, str(latest_model))
            self.model.eval()

            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def load_data(self):
        """Load data"""
        # Load timeline data
        self.timeline_df = pd.read_csv("data/reference/timeline.csv")
        print("Timeline data:")
        print(self.timeline_df)

        # Load simulated data (only take inspect=0 data as incoming ship pool)
        self.simulated_df = pd.read_csv("data/simulated/simulated_data.csv")

        # Only keep uninspected ships (inspect=0)
        self.ship_pool = self.simulated_df[self.simulated_df["inspect"] == 0].copy()
        print(f"Available ship pool size: {len(self.ship_pool)}")

        return True

    def distribute_yearly_to_monthly(self, yearly_value, year_seed=42):
        """Distribute yearly data into 12 months"""
        np.random.seed(year_seed)

        base_monthly = yearly_value // 12
        remainder = yearly_value % 12

        monthly_values = [base_monthly] * 12

        for i in range(remainder):
            monthly_values[i] += 1

        for i in range(12):
            max_increase = min(
                20, yearly_value - sum(monthly_values) + monthly_values[i]
            )
            max_decrease = min(20, monthly_values[i])

            adjustment = np.random.randint(-max_decrease, max_increase + 1)
            monthly_values[i] += adjustment

        current_sum = sum(monthly_values)
        if current_sum != yearly_value:
            diff = yearly_value - current_sum
            adjust_month = np.random.randint(0, 12)
            monthly_values[adjust_month] += diff

        return monthly_values

    def create_monthly_timeline(self):
        """Create monthly timeline data"""
        monthly_data = []

        for _, row in self.timeline_df.iterrows():
            year = int(row["year"])
            ships_in = int(row["number_ship_in"])
            ships_checked = int(row["number_checked"])

            # Distribute yearly data (number of incoming ships, number of inspected ships) into 12 months
            monthly_ships_in = self.distribute_yearly_to_monthly(
                ships_in, year_seed=year
            )
            monthly_ships_checked = self.distribute_yearly_to_monthly(
                ships_checked, year_seed=year + 1000
            )

            for month in range(1, 13):
                monthly_data.append(
                    {
                        "year": year,
                        "month": month,
                        "ships_in": monthly_ships_in[month - 1],
                        "ships_checked": monthly_ships_checked[month - 1],
                        "inspection_rate": monthly_ships_checked[month - 1]
                        / monthly_ships_in[month - 1]
                        if monthly_ships_in[month - 1] > 0
                        else 0,
                    }
                )

        self.monthly_df = pd.DataFrame(monthly_data)
        print(f"Created monthly data: {len(self.monthly_df)} months")
        return self.monthly_df

    def sample_ships_for_month(self, year, month, num_ships):
        """Sample ships for a specific month, ensuring no duplicates"""
        if not hasattr(self, "used_ship_ids"):
            self.used_ship_ids = set()

        num_ships = int(num_ships)

        # Sample from unused ships
        available_ships = self.ship_pool[~self.ship_pool["ID"].isin(self.used_ship_ids)]

        if len(available_ships) < num_ships:
            print(
                f"Warning: Year {year} Month {month} needs {num_ships} ships, but only {len(available_ships)} available"
            )
            num_ships = len(available_ships)

        # Random sampling
        sampled_ships = available_ships.sample(
            n=num_ships, random_state=int(year * 100 + month)
        )

        # Mark as used
        self.used_ship_ids.update(sampled_ships["ID"].tolist())

        return sampled_ships

    def nir_selection_method(self, ships_df, num_to_select):
        """NIR method to select ships (based on formal NIR risk assessment)"""
        ships_copy = ships_df.copy()

        try:
            # Use formal NIR method for risk assessment
            # First add ship information (ship_type, ship_age, num_deficiencies_36m, num_detention_36m, recognized_org_performance, company_performance, bgw_list)
            ships_with_info = add_shipping_info(ships_copy)

            # Execute NIR prediction and risk classification
            ships_with_nir = nir_predict(ships_with_info)

            # Prioritize selection based on NIR risk classification
            # Priority: HRS (High Risk) > SRS (Standard Risk) > LRS (Low Risk)
            risk_priority = {"HRS": 3, "SRS": 2, "LRS": 1}
            ships_with_nir["risk_priority"] = ships_with_nir["risk_category"].map(
                risk_priority
            )

            # Within the same risk level, sort by weighting_point
            ships_sorted = ships_with_nir.sort_values(
                ["risk_priority", "weighting_point"], ascending=[False, False]
            )

            # Select the top num_to_select ships
            selected = ships_sorted.head(num_to_select)

            print(
                f"NIR Selection: HRS={len(selected[selected['risk_category'] == 'HRS'])}, "
                f"SRS={len(selected[selected['risk_category'] == 'SRS'])}, "
                f"LRS={len(selected[selected['risk_category'] == 'LRS'])}"
            )

            return selected

        except Exception as e:
            print(f"NIR method failed, using simplified scoring: {e}")
            # Fallback to simplified NIR scoring logic
            ships_copy["nir_score"] = (
                ships_copy["NoDef"] * 0.3
                + ships_copy["DetInsp"] * 0.25
                + ships_copy["NoInsp"] * 0.2
                + ships_copy["InitialInsp"] * 0.15
                + (2024 - ships_copy["YOB"]) * 0.1  # Ship age factor
            )

            # Select ships with highest scores
            selected = ships_copy.nlargest(num_to_select, "nir_score")
            return selected

    def model_prediction_method(self, ships_df, num_to_select):
        """Model prediction method to select ships"""
        if self.model is None or self.processor is None:
            print("Model not loaded, using random selection")
            return ships_df.sample(n=num_to_select)

        try:
            # Preprocess data
            df_processed = self.processor.preprocess_data(ships_df, data_type="test")
            X_test, y_test = self.processor.transform_test_data(df_processed)

            # Model prediction
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                predictions = self.model(X_tensor)
                risk_scores = torch.softmax(predictions, dim=1)

                # Calculate composite risk score (standard risk*1 + high risk*2)
                composite_scores = risk_scores[:, 1] + risk_scores[:, 2] * 2

            # Add prediction scores to DataFrame
            ships_copy = ships_df.copy()
            ships_copy["model_risk_score"] = composite_scores.numpy()
            ships_copy["predicted_class"] = torch.argmax(predictions, dim=1).numpy()

            # Select ships with highest risk scores
            selected = ships_copy.nlargest(num_to_select, "model_risk_score")
            return selected

        except Exception as e:
            print(f"Model prediction failed: {e}")
            return ships_df.sample(n=num_to_select)

    def calculate_performance_metrics(self, selected_ships):
        """Calculate performance metrics"""
        total_checked = len(selected_ships)

        # Unqualified count (qualified=0)
        unqualified = len(selected_ships[selected_ships["qualified"] == 0])

        # Detained count (detained=1)
        detained = len(selected_ships[selected_ships["detained"] == 1])

        # Calculate rates
        unqualified_rate = unqualified / total_checked if total_checked > 0 else 0
        detention_rate = detained / total_checked if total_checked > 0 else 0

        return {
            "total_checked": total_checked,
            "unqualified": unqualified,
            "detained": detained,
            "unqualified_rate": unqualified_rate,
            "detention_rate": detention_rate,
        }

    def run_analysis(self):
        """Execute complete analysis"""
        print("Starting timeline analysis...")

        if not self.load_data():
            return False

        # Load model
        model_loaded = self.load_model_and_processor()
        if not model_loaded:
            print(
                "Warning: Model loading failed, using random selection instead of model prediction"
            )

        # Create monthly timeline
        monthly_df = self.create_monthly_timeline()

        # Store analysis results
        results = []

        # Reset used ship IDs
        self.used_ship_ids = set()

        print("Starting monthly analysis...")
        for _, row in monthly_df.iterrows():
            year = int(row["year"])
            month = int(row["month"])
            ships_in = int(row["ships_in"])
            ships_to_check = int(row["ships_checked"])

            print(
                f"Analyzing {year}-{month:02d}: {ships_in} ships entering, {ships_to_check} to inspect"
            )

            # Sample ships entering port for this month
            monthly_ships = self.sample_ships_for_month(year, month, ships_in)

            if len(monthly_ships) == 0:
                continue

            # Ensure inspection count doesn't exceed ships entering port
            actual_check_count = min(ships_to_check, len(monthly_ships))

            # NIR method selection
            nir_selected = self.nir_selection_method(monthly_ships, actual_check_count)
            nir_metrics = self.calculate_performance_metrics(nir_selected)

            # Model prediction method selection
            model_selected = self.model_prediction_method(
                monthly_ships, actual_check_count
            )
            model_metrics = self.calculate_performance_metrics(model_selected)

            # Store results
            results.append(
                {
                    "year": year,
                    "month": month,
                    "ships_in": ships_in,
                    "ships_checked": actual_check_count,
                    "inspection_rate": row["inspection_rate"],
                    "nir_unqualified_rate": nir_metrics["unqualified_rate"],
                    "nir_detention_rate": nir_metrics["detention_rate"],
                    "model_unqualified_rate": model_metrics["unqualified_rate"],
                    "model_detention_rate": model_metrics["detention_rate"],
                    "nir_unqualified": nir_metrics["unqualified"],
                    "nir_detained": nir_metrics["detained"],
                    "model_unqualified": model_metrics["unqualified"],
                    "model_detained": model_metrics["detained"],
                }
            )

        self.results_df = pd.DataFrame(results)

        self.save_results()

        # Generate charts
        self.create_visualizations()

        print("Analysis complete!")
        return True

    def save_results(self):
        """Save analysis results to CSV files"""
        # Inspection rates
        inspection_rate_df = self.results_df[
            ["year", "month", "ships_in", "ships_checked", "inspection_rate"]
        ].copy()
        inspection_rate_df.to_csv(
            "outputs/plots/monthly_inspection_rates.csv", index=False
        )

        # Unqualified rates
        unqualified_rate_df = self.results_df[
            [
                "year",
                "month",
                "ships_checked",
                "nir_unqualified_rate",
                "model_unqualified_rate",
                "nir_unqualified",
                "model_unqualified",
            ]
        ].copy()
        unqualified_rate_df.to_csv(
            "outputs/plots/monthly_unqualified_rates.csv", index=False
        )

        # Detention rates
        detention_rate_df = self.results_df[
            [
                "year",
                "month",
                "ships_checked",
                "nir_detention_rate",
                "model_detention_rate",
                "nir_detained",
                "model_detained",
            ]
        ].copy()
        detention_rate_df.to_csv(
            "outputs/plots/monthly_detention_rates.csv", index=False
        )

        print("Results saved to:")
        print("- monthly_inspection_rates.csv")
        print("- monthly_unqualified_rates.csv")
        print("- monthly_detention_rates.csv")

    def create_visualizations(self):
        """Create visualization charts"""
        # Set font for Chinese characters (if needed)
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        # Create time labels
        self.results_df["time_label"] = (
            self.results_df["year"].astype(str)
            + "-"
            + self.results_df["month"].astype(str).str.zfill(2)
        )

        # 1. Inspection rate chart
        plt.figure(figsize=(24, 5))

        time_labels = self.results_df["time_label"]
        label_indices = range(0, len(time_labels), 6)
        selected_labels = [time_labels.iloc[i] for i in label_indices]

        plt.plot(
            self.results_df["time_label"],
            self.results_df["inspection_rate"] * 100,
            marker="o",
            linewidth=2,
            markersize=6,
        )
        plt.title("Monthly Inspection Rate (%)", fontsize=16, fontweight="bold")
        plt.xlabel("Year-Month", fontsize=12)
        plt.ylabel("Inspection Rate (%)", fontsize=12)

        plt.xticks(selected_labels, rotation=45, fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            "outputs/plots/monthly_inspection_rate_timeline.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # 2. Unqualified rate comparison + grouped bar chart
        fig, ax1 = plt.subplots(1, 1, figsize=(24, 7))

        # Prepare data
        x_positions = np.arange(len(self.results_df))
        bar_width = 0.25

        ships_in = self.results_df["ships_in"].to_numpy()
        nir_unqualified = self.results_df["nir_unqualified"].to_numpy()
        model_unqualified = self.results_df["model_unqualified"].to_numpy()

        # Right Y-axis: unqualified rate line chart
        ax2 = ax1.twinx()
        ax2.plot(
            x_positions,
            self.results_df["nir_unqualified_rate"] * 100,
            "o-",
            label="NIR Unqualified Rate (%)",
            linewidth=3,
            markersize=6,
            color="#ff7f0e",
            zorder=1,
        )
        ax2.plot(
            x_positions,
            self.results_df["model_unqualified_rate"] * 100,
            "s-",
            label="Model Unqualified Rate (%)",
            linewidth=3,
            markersize=6,
            color="#2ca02c",
            zorder=1,
        )
        ax2.set_ylabel(
            "Unqualified Rate (%)", fontsize=12, fontweight="bold", color="red"
        )
        ax2.tick_params(axis="y", labelcolor="red")

        # Left Y-axis: ship count bar chart
        ax1.bar(
            x_positions - bar_width,
            ships_in,
            bar_width,
            label="Ships Entering Port",
            color="#1f77b4",
            alpha=0.7,
            zorder=2,
        )
        ax1.bar(
            x_positions,
            nir_unqualified,
            bar_width,
            label="NIR Unqualified Predictions",
            color="#ff7f0e",
            alpha=0.7,
            zorder=2,
        )
        ax1.bar(
            x_positions + bar_width,
            model_unqualified,
            bar_width,
            label="Model Unqualified Predictions",
            color="#2ca02c",
            alpha=0.7,
            zorder=2,
        )

        ax1.set_xlabel("Year-Month", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Number of Ships", fontsize=12, fontweight="bold", color="black")
        ax1.set_xticks(x_positions[::6])
        ax1.set_xticklabels(
            [
                self.results_df["time_label"].iloc[i]
                for i in range(0, len(self.results_df), 6)
            ],
            rotation=45,
            fontsize=10,
        )
        ax1.grid(True, alpha=0.3, zorder=0)

        # Merge legends and place in foreground
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(
            h1 + h2,
            l1 + l2,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            framealpha=1.0,
            fancybox=True,
            shadow=True,
        )
        legend.set_zorder(100)

        plt.title(
            "Monthly Unqualified Rate Comparison and Ship Numbers",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            "outputs/plots/monthly_unqualified_rate_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # 3. Detention rate comparison + grouped bar chart
        fig, ax1 = plt.subplots(1, 1, figsize=(24, 7))

        # Prepare data
        ships_checked = self.results_df["ships_checked"].to_numpy()
        nir_detained = self.results_df["nir_detained"].to_numpy()
        model_detained = self.results_df["model_detained"].to_numpy()

        # Right Y-axis: detention rate line chart
        ax2 = ax1.twinx()
        ax2.plot(
            x_positions,
            self.results_df["nir_detention_rate"] * 100,
            "o-",
            label="NIR Detention Rate (%)",
            linewidth=3,
            markersize=6,
            color="#ff7f0e",
            zorder=1,
        )
        ax2.plot(
            x_positions,
            self.results_df["model_detention_rate"] * 100,
            "s-",
            label="Model Detention Rate (%)",
            linewidth=3,
            markersize=6,
            color="#2ca02c",
            zorder=1,
        )
        ax2.set_ylabel(
            "Detention Rate (%)", fontsize=12, fontweight="bold", color="red"
        )
        ax2.tick_params(axis="y", labelcolor="red")

        # Left Y-axis: ship count bar chart
        ax1.bar(
            x_positions - bar_width,
            ships_checked,
            bar_width,
            label="Ships Inspected",
            color="#1f77b4",
            alpha=0.7,
            zorder=2,
        )
        ax1.bar(
            x_positions,
            nir_detained,
            bar_width,
            label="NIR Detention Predictions",
            color="#ff7f0e",
            alpha=0.7,
            zorder=2,
        )
        ax1.bar(
            x_positions + bar_width,
            model_detained,
            bar_width,
            label="Model Detention Predictions",
            color="#2ca02c",
            alpha=0.7,
            zorder=2,
        )

        ax1.set_xlabel("Year-Month", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Number of Ships", fontsize=12, fontweight="bold", color="black")
        ax1.set_xticks(x_positions[::6])
        ax1.set_xticklabels(
            [
                self.results_df["time_label"].iloc[i]
                for i in range(0, len(self.results_df), 6)
            ],
            rotation=45,
            fontsize=10,
        )
        ax1.grid(True, alpha=0.3, zorder=0)

        # Merge legends and place in foreground
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(
            h1 + h2,
            l1 + l2,
            loc="upper left",
            framealpha=1.0,
            fancybox=True,
            shadow=True,
        )
        legend.set_zorder(100)

        plt.title(
            "Monthly Detention Rate Comparison and Ship Numbers",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            "outputs/plots/monthly_detention_rate_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # 4. NIR and Model detention percentage comparison (single year analysis)
        fig, ax1 = plt.subplots(1, 1, figsize=(24, 7))

        first_year = self.results_df["year"].max()
        year_data = self.results_df[self.results_df["year"] == first_year].copy()

        year_data = year_data.sort_values("month")

        # Calculate cumulative data for the year
        year_data["cumulative_checked"] = year_data["ships_checked"].cumsum()
        year_data["cumulative_ships_in"] = year_data["ships_in"].cumsum()
        year_data["nir_percentage_detained"] = (
            year_data["nir_detained"].cumsum() / year_data["cumulative_checked"]
        ) * 100
        year_data["model_percentage_detained"] = (
            year_data["model_detained"].cumsum() / year_data["cumulative_checked"]
        ) * 100

        year_data["nir_cumulative_detained"] = year_data["nir_detained"].cumsum()
        year_data["model_cumulative_detained"] = year_data["model_detained"].cumsum()

        # Create month labels (1-12)
        month_labels = [f"Month {int(month)}" for month in year_data["month"]]
        x_positions_single_year = np.arange(len(year_data))

        # NIR and Model detention percentages
        ax1.plot(
            x_positions_single_year,
            year_data["nir_percentage_detained"],
            "o-",
            linewidth=3,
            markersize=6,
            color="#ff7f0e",
            label="NIR Percentage Detained (%)",
        )
        ax1.plot(
            x_positions_single_year,
            year_data["model_percentage_detained"],
            "s-",
            linewidth=3,
            markersize=6,
            color="#2ca02c",
            label="Model Percentage Detained (%)",
        )
        ax1.set_xlabel("Month", fontsize=12, fontweight="bold")
        ax1.set_ylabel(
            "Percentage of Ships Detained (%)", fontsize=12, fontweight="bold"
        )
        ax1.set_xticks(x_positions_single_year)
        ax1.set_xticklabels(month_labels, rotation=0, fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left", framealpha=1.0, fancybox=True, shadow=True)

        plt.title(
            f"{first_year} Monthly Detention Percentage Comparison",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            "outputs/plots/monthly_detention_percentage_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # 5. NIR and Model cumulative detention count comparison (single year analysis)
        fig, ax1 = plt.subplots(1, 1, figsize=(24, 7))

        # Cumulative detention counts
        ax1.plot(
            x_positions_single_year,
            year_data["nir_cumulative_detained"],
            "o-",
            linewidth=3,
            markersize=6,
            color="#ff7f0e",
            label="NIR Cumulative Detentions",
        )
        ax1.plot(
            x_positions_single_year,
            year_data["model_cumulative_detained"],
            "s-",
            linewidth=3,
            markersize=6,
            color="#2ca02c",
            label="Model Cumulative Detentions",
        )
        ax1.set_xlabel("Month", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Total Number of Detentions", fontsize=12, fontweight="bold")
        ax1.set_xticks(x_positions_single_year)
        ax1.set_xticklabels(month_labels, rotation=0, fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left", framealpha=1.0, fancybox=True, shadow=True)

        plt.title(
            f"{first_year} Monthly Cumulative Detentions Comparison",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            "outputs/plots/monthly_cumulative_detentions_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print("Charts generated:")
        print("- monthly_inspection_rate_timeline.png")
        print("- monthly_unqualified_rate_comparison.png")
        print("- monthly_detention_rate_comparison.png")
        print("- monthly_detention_percentage_comparison.png")
        print("- monthly_cumulative_detentions_comparison.png")


if __name__ == "__main__":
    analyzer = TimelineAnalyzer()
    analyzer.run_analysis()
