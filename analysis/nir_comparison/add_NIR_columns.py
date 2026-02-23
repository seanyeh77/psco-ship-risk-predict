import numpy as np
import pandas as pd
from typing import Optional


def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Data loaded into a DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def add_shipping_info(df: pd.DataFrame, _original_counts=None) -> pd.DataFrame:
    """
    Add shipping information to the DataFrame. （ship_type, ship_age, num_deficiencies_36m, num_detention_36m, recognized_org_performance, company_performance, bgw_list）

    Parameters:
    data (pd.DataFrame): DataFrame containing the main data.
    shipping_info (dict): Dictionary containing shipping information.

    Returns:
    pd.DataFrame: DataFrame with shipping information added.
    """
    original_counts = _original_counts or {
        "General cargo ship": 87,
        "Bulk carrier": 17,
        "Container ship": 64,
        "Oil tanker": 15,
        "Passenger ship": 85,
        "Chemical tanker": 20,
        "Gas carrier": 20,
        "Other": 7,
    }
    total_original = sum(original_counts.values())

    target_length = len(df)

    ship_counts_scaled = {
        ship: round(count / total_original * target_length)
        for ship, count in original_counts.items()
    }

    adjustment = target_length - sum(ship_counts_scaled.values())
    if adjustment != 0:
        max_type = max(ship_counts_scaled.items(), key=lambda x: x[1])[0]
        ship_counts_scaled[max_type] += adjustment

    ship_types = []
    for ship_type, count in ship_counts_scaled.items():
        ship_types.extend([ship_type] * count)

    np.random.shuffle(ship_types)

    df["ship_type"] = pd.Series(ship_types)

    # Calculate ship age
    current_year = pd.Timestamp.now().year
    df["ship_age"] = current_year - df["YOB"]

    # Calculate Number of deficiencies recorded in each inspection within previous 36 months
    # Formula: (NoDef / ship_age) * 3
    df["num_deficiencies_36m"] = ((df["NoDef"] / df["ship_age"]) * 3).fillna(0)

    # Calculate Number of detention within previous 36 months
    # Condition: Has detention and number of deficiencies in past 36 months
    df["num_detention_36m"] = np.where(
        (df["detained"] > 0), df["num_deficiencies_36m"].astype(int), 0
    )

    # Calculate Company performance based on yearly inspections
    yearly_inspections = df["NoInsp"] / df["ship_age"]
    yearly_inspections = yearly_inspections.fillna(0)

    # Categorize company performance based on yearly inspection rate
    # <0.1=Very Low, <=0.1=Low, >=0.2=High, else=Medium
    def categorize_company_performance(yearly_insp):
        if yearly_insp < 0.1:
            return "Very Low"
        elif yearly_insp <= 0.1:
            return "Low"
        elif yearly_insp >= 0.2:
            return "High"
        else:
            return "Medium"

    # Apply performance categorization to both recognized organization and company
    df["recognized_org_performance"] = yearly_inspections.apply(
        categorize_company_performance
    )
    df["company_performance"] = yearly_inspections.apply(categorize_company_performance)

    # Calculate BGW (Black-Grey-White) list status
    # Based on multiple criteria for risk assessment
    def calculate_bgw_status(row: pd.Series) -> str:
        black_criteria = []

        # Criterion 1: High-risk ship types
        high_risk_ship_types = [
            "Chemical tanker",
            "Gas carrier",
            "Oil tanker",
            "Bulk carrier",
            "Passenger ship",
            "Container ship",
        ]
        if row["ship_type"] in high_risk_ship_types:
            black_criteria.append(True)
        else:
            black_criteria.append(False)

        # Criterion 2: Ship age > 12 years
        if row["ship_age"] > 12:
            black_criteria.append(True)
        else:
            black_criteria.append(False)

        # Criterion 3: Poor recognized organization performance
        if row["recognized_org_performance"] in ["Low", "Very Low"]:
            black_criteria.append(True)
        else:
            black_criteria.append(False)

        # Criterion 4: Poor company performance or no inspections
        if row["company_performance"] in ["Low", "Very Low"] or row["NoInsp"] == 0:
            black_criteria.append(True)
        else:
            black_criteria.append(False)

        # Criterion 5: High number of deficiencies (>=5 in 36 months)
        if row["num_deficiencies_36m"] >= 5:
            black_criteria.append(True)
        else:
            black_criteria.append(False)

        # Criterion 6: Multiple detentions (>=3 in 36 months)
        if row["num_detention_36m"] >= 3:
            black_criteria.append(True)
        else:
            black_criteria.append(False)

        # Determine BGW status based on criteria met
        criteria_met = sum(black_criteria)

        if criteria_met >= 5:
            return "Black List"  # High risk - most criteria met
        elif criteria_met >= 3:
            return "Grey List"  # Medium risk - some criteria met
        else:
            return "White List"  # Low risk - few criteria met

    df["bgw_list"] = df.apply(calculate_bgw_status, axis=1)

    return df


def calculate_risk_category(row):
    """Determine risk category based on weighting points and specific criteria"""
    # HRS: High Risk Ships (weighting_point >= 4)
    if row["weighting_point"] >= 4:
        return "HRS"

    # LRS: Low Risk Ships (must meet ALL criteria)
    # - BGW-list is White AND
    # - Recognized Organization Performance is High AND
    # - Company performance is High AND
    # - All inspections have 5 or less deficiencies AND
    # - No detention
    elif (
        row["bgw_list"] == "White List"
        and row["recognized_org_performance"] == "High"
        and row["company_performance"] == "High"
        and row["num_deficiencies_36m"] <= 5
        and row["detained"] == 0
    ):
        return "LRS"

    # SRS: Standard Risk Ships (all other cases)
    else:
        return "SRS"


def nir_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate NIR (New Inspection Regime) risk assessment and weighting points
    """

    def calculate_weighting_point(row):
        """Calculate total weighting points based on various risk factors"""
        point = 0

        # Factor 1: High-risk ship types (+2 points)
        high_risk_ship_types = [
            "Chemical tanker",
            "Gas carrier",
            "Oil tanker",
            "Bulk carrier",
            "Passenger ship",
            "Container ship",
        ]
        if row["ship_type"] in high_risk_ship_types:
            point += 2
        else:
            point += 0

        # Factor 2: Ship age > 12 years (+1 point)
        if row["ship_age"] > 12:
            point += 1
        else:
            point += 0

        # Factor 3: Black list status (+1 point)
        if row["bgw_list"] == "Black List":
            point += 1
        else:
            point += 0

        # Factor 4: Poor recognized organization performance (+1 point)
        if row["recognized_org_performance"] in ["Low", "Very Low"]:
            point += 1
        else:
            point += 0

        # Factor 5: Poor company performance or no inspections (+2 points)
        if row["company_performance"] in ["Low", "Very Low"] or row["NoInsp"] == 0:
            point += 2
        else:
            point += 0

        # Factor 6: High deficiencies (>=5 in 36 months, add actual count)
        if row["num_deficiencies_36m"] >= 5:
            point += row["num_deficiencies_36m"]
        else:
            point += 0

        # Factor 7: Multiple detentions (>=3 in 36 months, +1 point)
        if row["num_detention_36m"] >= 3:
            point += 1
        else:
            point += 0

        return point

    # Apply weighting point calculation
    df["weighting_point"] = df.apply(calculate_weighting_point, axis=1)

    # Calculate risk classification using standalone function
    df["risk_category"] = df.apply(calculate_risk_category, axis=1)

    return df


def main():
    """
    Main function to execute the NIR analysis script.
    """
    df = load_data(input_file_path)
    if df is None:
        print("No data to process.")
        return

    # Add shipping information and calculated features
    df = add_shipping_info(df, original_counts)

    # # Perform NIR prediction and risk classification
    # df = nir_predict(df)

    # # Display result statistics
    # print("Risk Category Distribution:")
    # print(df['risk_category'].value_counts())
    print("\nBGW List Distribution:")
    print(df["bgw_list"].value_counts())

    df.to_csv(output_file_path, index=False)
    return df


# Original ship type distribution for scaling
original_counts = {
    "General cargo ship": 87,
    "Bulk carrier": 17,
    "Container ship": 64,
    "Oil tanker": 15,
    "Passenger ship": 85,
    "Chemical tanker": 20,
    "Gas carrier": 20,
    "Other": 7,
}

input_file_path = "data/simulated/simulated_data.csv"  # Input data file path
output_file_path = "data/processed/processed_data.csv"  # Output data file path

if __name__ == "__main__":
    main()
