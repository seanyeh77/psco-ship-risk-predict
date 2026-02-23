"""
Main training script for PSCO model
"""

import os
import sys
import logging
from typing import Dict, Union
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from psco.config import Config
from psco.data_processor import DataProcessor
from psco.model import PSCOModel, create_model
from psco.trainer import (
    PSCOTrainer,
    create_data_loaders,
    save_model,
    plot_confusion_matrix,
)


def setup_logging(config: Config):
    """Setup logging configuration"""
    os.makedirs(config.paths.logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.paths.logs_dir, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    return log_file


def main():
    """Main training pipeline"""
    config = Config()

    log_file = setup_logging(config)
    logging.info("=" * 50)
    logging.info("PSCO Model Training")
    logging.info("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logging.info(f"GPU count: {torch.cuda.device_count()}")
        logging.info(f"GPU name: {torch.cuda.get_device_name(0)}")

    try:
        # 1. Data processing
        logging.info("1. Processing data...")
        data_processor = DataProcessor(config)

        # Load data
        df = data_processor.load_data()
        logging.info(f"Raw data shape: {df.shape}")

        # Preprocess training data
        df_train = data_processor.preprocess_data(df, data_type="train")
        logging.info(f"Training data shape: {df_train.shape}")

        # Split training and validation data
        X_train, X_val, y_train, y_val = data_processor.split_data(df_train)

        # Log data distribution
        data_processor.log_data_distribution(y_train, "Training set")
        data_processor.log_data_distribution(y_val, "Validation set")

        # Compute class weights
        class_weights: Dict[int, float] = data_processor.compute_class_weights(y_train)

        # 2. Model creation
        logging.info("2. Creating model...")
        model: Union[PSCOModel, nn.DataParallel] = create_model(config)
        if isinstance(model, PSCOModel):
            model_info = model.get_model_info()
        elif isinstance(model, nn.DataParallel) and isinstance(model.module, PSCOModel):
            model_info = model.module.get_model_info()
        else:
            raise TypeError(
                f"Unexpected model type: {type(model)}. Ensure create_model returns a valid PSCOModel or nn.DataParallel instance."
            )

        logging.info("Model statistics:")
        for key, value in model_info.items():
            logging.info(f"  {key}: {value}")

        # 3. Data loaders
        logging.info("3. Creating data loaders...")
        train_loader, val_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, config
        )

        logging.info(f"Training batches: {len(train_loader)}")
        logging.info(f"Validation batches: {len(val_loader)}")

        # 4. Trainer setup
        logging.info("4. Setting up trainer...")
        trainer = PSCOTrainer(model, config, class_weights, device)

        # 5. Training
        logging.info("5. Starting training...")
        training_results = trainer.train(train_loader, val_loader)

        # 6. Save model
        logging.info("6. Saving model...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(
            config.paths.models_dir, f"psco_model_{timestamp}.pth"
        )

        save_model(
            model=model,
            config=config,
            save_path=model_path,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            epoch=len(training_results["train_losses"]),
            metrics=training_results,
        )

        # 7. Plot training history
        logging.info("7. Plotting training history...")
        plot_path = os.path.join(
            config.paths.plots_dir, f"training_history_{timestamp}.png"
        )
        trainer.plot_training_history(plot_path)

        # 8. Final evaluation
        logging.info("8. Final evaluation...")
        eval_results = trainer.evaluate(val_loader)

        logging.info(f"Final validation accuracy: {eval_results['accuracy']:.2f}%")

        # 9. Classification report
        class_names = data_processor.get_class_names()
        report = classification_report(
            eval_results["targets"],
            eval_results["predictions"],
            target_names=class_names,
            digits=4,
        )
        logging.info(f"Classification report:\\n{report}")

        # 10. Confusion matrix
        logging.info("10. Plotting confusion matrix...")
        cm_path = os.path.join(
            config.paths.plots_dir, f"confusion_matrix_{timestamp}.png"
        )
        plot_confusion_matrix(
            eval_results["targets"], eval_results["predictions"], class_names, cm_path
        )

        # 11. Save data processor state
        logging.info("11. Saving data processor state...")
        processor_path = os.path.join(
            config.paths.models_dir, f"data_processor_{timestamp}.pth"
        )
        torch.save(
            {
                "scaler": data_processor.scaler,
                "class_weights": data_processor.class_weights,
                "config": config,
            },
            processor_path,
        )

        logging.info("=" * 50)
        logging.info("Training completed!")
        logging.info("=" * 50)
        logging.info(f"Model saved to: {model_path}")
        logging.info(f"Data processor saved to: {processor_path}")
        logging.info(f"Training history plot: {plot_path}")
        logging.info(f"Confusion matrix plot: {cm_path}")
        logging.info(f"Log file: {log_file}")

        return {
            "model_path": model_path,
            "processor_path": processor_path,
            "results": training_results,
            "eval_results": eval_results,
        }

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback

        logging.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
