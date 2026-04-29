# Rossmann Sales Forecasting Project

Sales forecasting project for Rossmann stores using machine learning models (XGBoost, CatBoost) and hierarchical time series analysis.

## Project Structure

```
├── Clean_notebooks/          # Jupyter notebooks for data processing and modeling
│   ├── 01_data_cleaning.ipynb
│   ├── 02_sql_features.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_xgboost_model.ipynb
│   ├── 05_catboost.ipynb
│   ├── 06_hierarchical_forecasting.ipynb
│   └── 07_anomaly_and_backtesting.ipynb
│
├── Dataset/                  # Original datasets
│   ├── store.csv
│   ├── store_states.csv
│   ├── train.csv
│   └── test.csv
│
├── Intermediate_files/       # Generated files during analysis
│   ├── cleaned_rossmann.csv
│   ├── rossmann_features.csv
│   ├── sql_features.csv
│   ├── xgb_predictions.csv
│   ├── final_predictions.csv
│   └── ...
│
├── README.md                 # This file
└── .gitignore               # Git ignore rules
```

## Notebooks Overview

1. **Data Cleaning** - Handle missing values, outliers, and data quality issues
2. **SQL Features** - Generate features using SQL-based aggregations
3. **Feature Engineering** - Create time-based and store-based features
4. **XGBoost Model** - Train and evaluate XGBoost forecasting model
5. **CatBoost Model** - Train and evaluate CatBoost forecasting model
6. **Hierarchical Forecasting** - Implement hierarchical time series reconciliation
7. **Anomaly & Backtesting** - Detect anomalies and validate model performance

## Models Used

- XGBoost
- CatBoost
- Hierarchical Time Series Forecasting

## Results

- Final predictions stored in `Intermediate_files/final_predictions.csv`
- Model performance metrics in respective notebooks

## Requirements

- Python 3.x
- Jupyter Notebook
- XGBoost
- CatBoost
- Pandas, NumPy, Scikit-learn