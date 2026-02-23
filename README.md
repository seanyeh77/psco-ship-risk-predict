# PSCO - Port State Control Officer Prediction System

A machine learning system for predicting vessel detention outcomes using historical inspection data.

## 📁 Project Structure

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

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
cd /path/to/psco
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package in editable mode:
```bash
pip install -e .
```

### Usage

#### Training a Model
```bash
python scripts/train.py
```

#### Evaluating a Model
```bash
python scripts/evaluate.py
```

#### Generating Predictions
```bash
python analysis/nir_comparison/generate_predictions.py
```

#### Analyzing Results
```bash
python analysis/exploratory/analyze_predictions.py
```

## 📊 Features

- **Neural Network Model**: Custom PyTorch-based model for detention prediction
- **Feature Engineering**: Comprehensive data processing and feature extraction
- **NIR Comparison**: Compare model predictions with NIR (New Inspection Regime) system
- **Timeline Analysis**: Temporal analysis of inspection and detention trends
- **Feature Importance**: Multiple methods for analyzing feature importance

## 🔧 Configuration

Model configuration is managed in `src/psco/config.py`. Key settings include:

- Model architecture parameters
- Training hyperparameters
- Data processing options
- File paths and directories

## 📝 Development

### Running Tests
```bash
pytest tests/
```

### Code Structure
- All core functionality is in `src/psco/`
- Scripts for training/evaluation are in `scripts/`
- Analysis notebooks and scripts are in `analysis/`
- Tests follow the source structure in `tests/`

## 📈 Results

Model outputs are saved to:
- Trained models: `models/`
- Predictions: `outputs/predictions/`
- Visualizations: `outputs/plots/`
- Logs: `logs/`

## 📚 Documentation

Additional documentation can be found in the `docs/` directory.

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests to ensure everything works
4. Submit a pull request

## 📄 License

[Add your license information here]

## 👥 Authors

[Add author information here]