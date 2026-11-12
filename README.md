# House Price Prediction

A supervised machine-learning regression project that estimates residential sale prices from property characteristics. The workflow demonstrates preprocessing, a train/test split, baseline comparison and regression evaluation with scikit-learn.

> **Dataset note:** The included housing data is synthetic and is suitable for code demonstration only—not property valuation, lending or investment decisions.

## Project objectives

- Explore a structured housing dataset.
- Prepare numeric and categorical features.
- Split data into training and test sets.
- Compare a mean-value baseline, Linear Regression and Random Forest.
- Evaluate predictions using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and R-squared.

## Repository structure

```text
.
├── data/
│   └── housing.csv
├── src/
│   └── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

Running the training script creates an `artifacts/` folder locally. The evaluation report is tracked at [artifacts/metrics.json](artifacts/metrics.json); the serialized model file is excluded from Git.

## Evaluation snapshot

Results from the committed run:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline | 164,545.45 | 182,734.43 | -4.286 |
| Linear Regression | **10,446.43** | **14,075.75** | **0.969** |
| Random Forest | 30,860.65 | 34,441.17 | 0.812 |

**Best model in this run: Linear Regression.** Error values are expressed in the target variable's dataset units; the dataset does not establish a real-world currency or market.

The test-split R² of approximately 0.969 indicates a close fit on this particular split. Because the dataset is synthetic and small, this should not be interpreted as evidence of reliable performance on real housing markets. Validate with more representative data, cross-validation and residual analysis before making broader claims.

## Getting started

### Requirements

- Python 3.10 or newer
- pip

### Windows — Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python src/train.py
```

### Windows — PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

## Tools and skills demonstrated

- Python, pandas and NumPy
- scikit-learn preprocessing and regression pipelines
- Linear Regression and Random Forest
- Baseline modeling and train/test evaluation
- MAE, RMSE and R-squared
- Reproducible experiments and Git

## Limitations and next steps

- Synthetic data does not capture the complexity of actual housing markets.
- A single small test split can produce unstable metrics.
- The model is not deployed and should not be used to make property decisions.
- Useful extensions include cross-validation, residual plots, feature-importance analysis and evaluation on a larger, documented public dataset.
