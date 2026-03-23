# PSCO - Port State Control Officer Prediction System

A machine learning system for predicting vessel detention outcomes using historical inspection data.

## Project Structure

```
psco/
├── src/psco/              # Core package modules
│   ├── config.py          # Configuration management
│   ├── data_processor.py  # Data processing and feature engineering
│   ├── model.py           # Neural network model definition
│   ├── trainer.py         # Training and evaluation logic
│   └── feature_importance.py  # Feature importance analysis
│
├── scripts/               # Executable scripts
│   ├── train.py          # Train the PSCO model
│   ├── evaluate.py       # Evaluate trained model
│   ├── generate_data.py  # Generate simulated data
│   └── feature_analysis.py  # Analyze feature importance
│
├── analysis/              # Analysis scripts
│   ├── nir_comparison/   # NIR system comparison
│   │   ├── add_NIR_columns.py
│   │   ├── model_comparison.py
│   │   └── generate_predictions.py
│   └── exploratory/      # Exploratory analysis
│       ├── analyze_predictions.py
│       ├── timeline_analysis.py
│       └── timeline_summary.py
│
├── tests/                 # Test files
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── test_simulator.py
│
├── data/                  # Data files
│   ├── raw/              # Raw input data
│   ├── processed/        # Processed datasets
│   ├── reference/        # Reference data (timeline, summaries)
│   └── simulated/        # Simulated data
│
├── models/                # Saved model files
├── outputs/               # Generated outputs
│   ├── predictions/      # Prediction results
│   ├── plots/           # Visualization plots
│   └── reports/         # Analysis reports
│
├── logs/                  # Log files
├── docs/                  # Documentation
└── configs/               # Configuration files
```

## Quick Start

### Prerequisites

Before you begin, ensure you have uv installed on your system. uv is an extremely fast Python package installer and resolver.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/seanyeh77/psco-ship-risk-predict.git
```

2. Install the package in editable mode with UV:
```bash
uv pip install -e .
```

UV will automatically create and manage the virtual environment for you - no need to manually activate it!

### Usage

#### Training a Model
```bash
uv run python scripts/train.py
```

#### Evaluating a Model
```bash
uv run python scripts/evaluate.py
```

#### Generating Predictions
```bash
uv run python analysis/nir_comparison/generate_predictions.py
```

#### Analyzing Results
```bash
uv run python analysis/exploratory/analyze_predictions.py
```

## Features

- **Neural Network Model**: Custom PyTorch-based model for detention prediction
- **Feature Engineering**: Comprehensive data processing and feature extraction
- **NIR Comparison**: Compare model predictions with NIR (New Inspection Regime) system
- **Timeline Analysis**: Temporal analysis of inspection and detention trends
- **Feature Importance**: Multiple methods for analyzing feature importance

## Configuration

Model configuration is managed in `src/psco/config.py`. Key settings include:

- Model architecture parameters
- Training hyperparameters
- Data processing options
- File paths and directories

## Development

### Running Tests
```bash
uv run pytest tests/
```

### Code Quality
```bash
# Linting
uv run ruff check

# Formatting
uv run ruff format
```

### Code Structure
- All core functionality is in `src/psco/`
- Scripts for training/evaluation are in `scripts/`
- Analysis notebooks and scripts are in `analysis/`
- Tests follow the source structure in `tests/`

## Results

Model outputs are saved to:
- Trained models: `models/`
- Predictions: `outputs/predictions/`
- Visualizations: `outputs/plots/`
- Logs: `logs/`

## Documentation

Additional documentation can be found in the `docs/` directory.