"""
Generate predictions using PSCO model and NIR system
Save results to outputs/predictions/ directory for further analysis
"""

import os
import warnings
from typing import Optional
from datetime import datetime

import pandas as pd
import torch

from psco.config import Config
from psco.data_processor import DataProcessor
from psco.model import create_model
from psco.trainer import load_model

warnings.filterwarnings("ignore")


def load_psco_model_and_predict(
    data_path: str,
    model_path: Optional[str] = None,
    processor_path: Optional[str] = None,
):
    """
    Load PSCO model and make predictions
    """
    print("Loading PSCO model...")

    config = Config()

    # Use latest model file if not specified
    if model_path is None:
        models_dir = config.paths.models_dir
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
        models_dir = config.paths.models_dir
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
            "psco_prob_0": probabilities[:, 0].cpu().numpy(),
            "psco_prob_1": probabilities[:, 1].cpu().numpy(),
            "psco_prob_2": probabilities[:, 2].cpu().numpy(),
            "psco_prediction": predictions.cpu().numpy(),
            "true_label": y_test,
        }
    )

    return results_df, test_data


def generate_nir_predictions(data_path: str):
    """
    Generate NIR prediction results using nir_predict function
    """
    print("Generating NIR predictions...")

    try:
        # Import the NIR module
        import add_NIR_columns

        # Load the data
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples for NIR prediction")

        # Add shipping information first
        df_with_shipping = add_NIR_columns.add_shipping_info(df)

        # Run NIR prediction
        nir_results = add_NIR_columns.nir_predict(df_with_shipping)
        print("NIR prediction completed successfully")

        # Convert NIR risk categories to numerical values
        nir_risk_map = {"LRS": 0, "SRS": 1, "HRS": 2}
        nir_results["nir_prediction"] = nir_results["risk_category"].map(nir_risk_map)

        return nir_results[["ID", "nir_prediction", "risk_category", "weighting_point"]]

    except ImportError as e:
        print(f"Error importing NIR module: {e}")
        raise
    except Exception as e:
        print(f"Error running NIR predictions: {e}")
        raise


def save_predictions(psco_results, nir_results, test_data, output_dir):
    """
    Save prediction results to data directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save PSCO predictions
    psco_output_path = os.path.join(output_dir, f"psco_predictions_{timestamp}.csv")
    psco_results.to_csv(psco_output_path, index=False)
    print(f"PSCO predictions saved to: {psco_output_path}")

    # Save NIR predictions
    nir_output_path = os.path.join(output_dir, f"nir_predictions_{timestamp}.csv")
    nir_results.to_csv(nir_output_path, index=False)
    print(f"NIR predictions saved to: {nir_output_path}")

    # Merge and save combined results
    merged_df = psco_results.merge(nir_results, on="ID", how="inner")

    # Add original features
    test_features = test_data[
        ["ID", "detained", "qualified", "YOB", "GT", "NoInsp", "NoDef"]
    ].copy()
    merged_df = merged_df.merge(test_features, on="ID", how="inner")

    # Save merged predictions
    merged_output_path = os.path.join(output_dir, f"merged_predictions_{timestamp}.csv")
    merged_df.to_csv(merged_output_path, index=False)
    print(f"Merged predictions saved to: {merged_output_path}")

    # Save metadata
    metadata = {
        "generation_time": timestamp,
        "psco_samples": len(psco_results),
        "nir_samples": len(nir_results),
        "merged_samples": len(merged_df),
        "psco_file": f"psco_predictions_{timestamp}.csv",
        "nir_file": f"nir_predictions_{timestamp}.csv",
        "merged_file": f"merged_predictions_{timestamp}.csv",
    }

    metadata_path = os.path.join(output_dir, f"prediction_metadata_{timestamp}.json")
    import json

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    return merged_df, timestamp


def main():
    """Main function"""
    try:
        print("Starting prediction generation...")

        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))

        # Data path
        data_path = os.path.join(
            project_root, "data", "simulated", "simulated_data.csv"
        )
        output_dir = os.path.join(project_root, "outputs", "predictions")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Generate PSCO model predictions
        psco_results, test_data = load_psco_model_and_predict(data_path)
        print(f"PSCO model prediction completed: {len(psco_results)} samples")

        # 2. Generate NIR predictions
        nir_results = generate_nir_predictions(data_path)
        print(f"NIR prediction generation completed: {len(nir_results)} samples")

        # 3. Save all predictions
        merged_df, timestamp = save_predictions(
            psco_results, nir_results, test_data, output_dir
        )
        print(f"All predictions saved with timestamp: {timestamp}")
        print(f"Merged dataset contains: {len(merged_df)} samples")

        print("\nPrediction generation completed successfully!")
        print(f"Files saved to: {output_dir}")

    except Exception as e:
        print(f"Error occurred during prediction generation: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
