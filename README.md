# AI Car Price Prediction Dashboard

A complete Streamlit project for car price prediction using machine learning. This application enables users to upload a CSV dataset, preprocess data, perform exploratory data analysis, train regression models, tune hyperparameters, and make manual predictions.

## Features

- CSV dataset upload with preview, shape, column list, and missing value summary
- Selectable preprocessing options: missing value handling, encoding, scaling, and outlier removal
- Interactive visualizations: correlation heatmap, histograms, boxplots, scatter plots, pairplots, and feature importance
- Multiple model choices: Linear Regression, Decision Tree, Random Forest, KNN, and XGBoost
- Train-test split controls and optional cross-validation
- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Model evaluation metrics and actual vs predicted visualization
- Manual prediction form for car details

## Installation

1. Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the Streamlit app:

```bash
streamlit run app.py
```

## Usage

1. Upload a CSV dataset containing car features and a target price column.
2. Select the target column and apply preprocessing options.
3. Explore data using the EDA section.
4. Choose and train a regression model.
5. Use the manual prediction section to estimate car prices.

## Notes

If your dataset uses different column names for car details, ensure the target column and feature mappings are inferred correctly in the preprocessing step.
