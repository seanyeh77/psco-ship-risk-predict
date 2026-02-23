import matplotlib.pyplot as plt
import pandas as pd
import logging
import os

# File paths
DETENTION_PATH = "data/reference/detention_summary.csv"
SIMULATED_PATH = "data/simulated/simulated_data.csv"

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("validation.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


def validate_sample_counts(df: pd.DataFrame) -> None:
    """
    Validate whether the number of samples by type matches expectations.
    """
    total = len(df)
    detained_fail = df[(df["qualified"] == 0) & (df["detained"] == 1)]
    undetained_fail = df[(df["qualified"] == 0) & (df["detained"] == 0)]
    passed = df[(df["qualified"] == 1) & (df["detained"] == 0)]

    logging.info("Sample count validation results:")
    logging.info(f"Total samples: {total} (Expected: 10000)")
    logging.info(
        f"Unqualified and detained samples: {len(detained_fail)} (Expected: 1000)"
    )
    logging.info(
        f"Unqualified but not detained samples: {len(undetained_fail)} (Expected: 3000)"
    )
    logging.info(f"Qualified and not detained samples: {len(passed)} (Expected: 6000)")


def validate_value_ranges(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """
    Validate whether variable values fall within expected ranges for different sample types,
    and check if their averages are close to theoretical means.
    """
    logging.info("Value range and average validation:")
    for _, row in summary.iterrows():
        variable = row["Variable"]

        groups = {
            "Unqualified and Detained": {
                "data": df[(df["qualified"] == 0) & (df["detained"] == 1)][variable],
                "min": row["Min_1"],
                "max": row["Max_1"],
                "mean": row["Mean_1"],
            },
            "Unqualified but Not Detained": {
                "data": df[(df["qualified"] == 0) & (df["detained"] == 0)][variable],
                "min": (row["Min_0"] + row["Min_1"]) / 2,
                "max": (row["Max_0"] + row["Max_1"]) / 2,
                "mean": (row["Mean_0"] + row["Mean_1"]) / 2,
            },
            "Qualified and Not Detained": {
                "data": df[(df["qualified"] == 1) & (df["detained"] == 0)][variable],
                "min": row["Min_0"],
                "max": row["Max_0"],
                "mean": row["Mean_0"],
            },
        }

        for label, info in groups.items():
            data = info["data"]
            min_expected = info["min"]
            max_expected = info["max"]
            mean_expected = info["mean"]

            min_actual = data.min()
            max_actual = data.max()
            mean_actual = data.mean()

            range_valid = (min_actual >= min_expected) and (max_actual <= max_expected)
            mean_valid = (
                abs(mean_actual - mean_expected) <= 0.05 * abs(mean_expected)
                if mean_expected != 0
                else True
            )

            logging.info(f"[{variable}] Category: {label}")
            logging.info(
                f"  Actual min: {min_actual:.3f}, Expected min: {min_expected:.3f}; "
                f"Actual max: {max_actual:.3f}, Expected max: {max_expected:.3f}; "
                f"Actual mean: {mean_actual:.3f}, Expected mean: {mean_expected:.3f}"
            )
            logging.info(
                f"  Range validation: {'PASS' if range_valid else 'FAIL'}, Mean validation: {'PASS' if mean_valid else 'FAIL'}"
            )


def plot_df(df: pd.DataFrame, column: str, title: str = "", file_name="temp") -> None:
    """
    Plot a histogram for the specified column.
    """

    plt.figure(figsize=(10, 6))
    df[column].hist(bins=30, edgecolor="black")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    # Show information on canvas
    plt.text(
        0.05,
        0.95,
        f"Min: {df[column].min():.2f}\nMax: {df[column].max():.2f}\nMean: {df[column].mean():.2f}\nStd: {df[column].std():.2f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig(f"outputs/plots/{file_name}.png", bbox_inches="tight", dpi=300)


# Plot histograms for all sections (unqualified and detained, unqualified but not detained, qualified and not detained)
def plot_all_columns(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """
    Plot histograms for all variables.
    """
    for _, row in summary.iterrows():
        variable = row["Variable"]
        plot_df(
            df[(df["qualified"] == 0) & (df["detained"] == 1)],
            variable,
            f"not qualified and detained - {variable}",
            f"not_qualified_and_detained_{variable}",
        )
        plot_df(
            df[(df["qualified"] == 0) & (df["detained"] == 0)],
            variable,
            f"not qualified and undetained - {variable}",
            f"not_qualified_and_undetained_{variable}",
        )
        plot_df(
            df[(df["qualified"] == 1) & (df["detained"] == 0)],
            variable,
            f"qualified and undetained - {variable}",
            f"qualified_and_undetained_{variable}",
        )


def main():
    """
    Validate sample counts and variable value reasonableness.
    """
    if not os.path.exists(DETENTION_PATH):
        logging.error(f"Reference file not found: {DETENTION_PATH}")
        return

    if not os.path.exists(SIMULATED_PATH):
        logging.error(f"Simulated data file not found: {SIMULATED_PATH}")
        return

    try:
        df_summary = pd.read_csv(DETENTION_PATH)
        df_simulated = pd.read_csv(SIMULATED_PATH)

        logging.info("Verify...\n")
        validate_sample_counts(df_simulated)
        logging.info("")
        validate_value_ranges(df_simulated, df_summary)
        logging.info("\nVerification finished.")
        logging.info("Plotting...\n")
        plot_all_columns(df_simulated, df_summary)
        logging.info("\nPlotting finished.")

    except Exception as e:
        logging.exception(f"Error: {e}")


if __name__ == "__main__":
    main()
