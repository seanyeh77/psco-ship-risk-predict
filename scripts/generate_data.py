# data_simulator.py

from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import uuid


def generate_truncated_normal_data(
    a, b, mu, sigma, size=10000, tol=1e-3, max_iter=100000
) -> pd.DataFrame:
    """
    Generate truncated normal distribution data.

    Args:
        a (float): Lower truncation limit
        b (float): Upper truncation limit
        mu (float): Mean of the normal distribution
        sigma (float): Standard deviation of the normal distribution
        size (int): Number of samples to generate

    Returns:
        pandas.DataFrame: DataFrame containing the generated data
    """

    a_, b_ = (a - mu) / sigma, (b - mu) / sigma

    loc = mu

    for _ in range(max_iter):
        a_, b_ = (a - loc) / sigma, (b - loc) / sigma
        dist = truncnorm(a_, b_, loc=loc, scale=sigma)
        samples = dist.rvs(size=size)
        actual_mean = samples.mean()

        error = actual_mean - mu
        if abs(error) < tol:
            break

        loc -= error * 0.5

    samples = np.clip(samples, a, b)
    return pd.DataFrame(samples, columns=["Value"])


def plot_df(df: pd.DataFrame, column: str, title: str = "") -> None:
    """
    Plot a histogram for a specified column in the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the data
        column (str): Name of the column to plot
    """
    print(
        f"Min: {df[column].min()}, Max: {df[column].max()}, Mean: {df[column].mean()}, Std: {df[column].std()}"
    )
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=50, density=True, alpha=0.6, color="g")
    # plt.title(f"Histogram of {column}")
    plt.title(title if title else f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.grid()
    # plt.show()


def read_datascope_file(file_path: str) -> pd.DataFrame:
    """
    Read a Datascope file and return a DataFrame.

    Args:
        file_path (str): Path to the Datascope file

    Returns:
        pandas.DataFrame: DataFrame containing the file data
    """
    df = pd.read_csv(file_path)
    return df


def add_UUID(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an ID column to the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame to add ID column to

    Returns:
        pandas.DataFrame: DataFrame with ID column added
    """

    df["ID"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df = df[["ID"] + [col for col in df.columns if col != "ID"]]
    return df


DETENTION_PATH = "data/reference/detention_summary.csv"
SIMULATED_PATH = "data/simulated/simulated_data.csv"


def create_data(
    total: int, unqualified_ratio: float, detained_ratio: float
) -> pd.DataFrame:

    unqualified = int(total * unqualified_ratio)
    detained = int(total * detained_ratio)
    # Create N records
    df = pd.DataFrame(
        {"inspect": [1] * total, "qualified": [1] * total, "detained": [0] * total}
    )

    # Unqualified count: 4000
    df.loc[0 : unqualified - 1, "qualified"] = 0

    # Unqualified and detained count: 1000
    df.loc[0 : detained - 1, "detained"] = 1

    # Generate truncated normal distribution data
    try:
        # Read detention_summary.csv
        df_detention = read_datascope_file(DETENTION_PATH)

        # Generate data for unqualified and detained cases
        for index, row in df_detention.iterrows():
            a = row["Min_1"]
            b = row["Max_1"]
            mu = row["Mean_1"]
            sigma = max(mu - a, b - mu) / 2
            df.loc[0 : detained - 1, row["Variable"]] = generate_truncated_normal_data(
                a, b, mu, sigma, size=detained
            )["Value"].values
        # # Plot histogram
        # for column in df_detention['Variable']:
        #     plot_df(df.loc[:999], column, title=f"Min:{a} Max:{b} Mean:{mu} Std:{sigma}")

        # Generate data for unqualified but not detained cases
        for index, row in df_detention.iterrows():
            a = (row["Min_0"] + row["Min_1"]) / 2
            b = (row["Max_0"] + row["Max_1"]) / 2
            mu = (row["Mean_0"] + row["Mean_1"]) / 2
            sigma = max(mu - a, b - mu) / 2
            df.loc[detained : unqualified - 1, row["Variable"]] = (
                generate_truncated_normal_data(
                    a, b, mu, sigma, size=(unqualified - detained)
                )["Value"].values
            )
        # # Plot histogram
        # for column in df_detention['Variable']:
        #     plot_df(df, column, title=f"Min:{a} Max:{b} Mean:{mu} Std:{sigma}")

        # Generate data for qualified and not detained cases
        for index, row in df_detention.iterrows():
            a = row["Min_0"]
            b = row["Max_0"]
            mu = row["Mean_0"]
            sigma = max(mu - a, b - mu) / 2
            df.loc[unqualified : total - 1, row["Variable"]] = (
                generate_truncated_normal_data(
                    a, b, mu, sigma, size=(total - unqualified)
                )["Value"].values
            )
        # # Plot histogram
        # for column in df_detention['Variable']:
        #     plot_df(df.loc[4000:9999], column, title=f"Min:{a} Max:{b} Mean:{mu} Std:{sigma}")

        # Save results to CSV
        return df

    except FileNotFoundError:
        print(f"File not found: {DETENTION_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return pd.DataFrame()


if __name__ == "__main__":
    df = create_data(50000, 0.2, 0.05)
    df["inspect"] = 0
    df2 = create_data(10000, 0.4, 0.1)
    df2["inspect"] = 1
    df = pd.concat([df, df2], ignore_index=True)
    df = add_UUID(df)
    df.to_csv(SIMULATED_PATH, index=False)
