# AML Project 3: Ensemble Model Comparison

## Overview
This project compares 5 classification models (logistic regression, random forest, gradient boosting, XGBoost, and stacking) using 5-fold cross-validation.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your data in `data/data.csv`

## Running the Pipeline

```bash
python main.py
```

This will:
- Load and preprocess the data
- Train all 5 models with 5-fold CV
- Save cross-validation folds and results
- Generate performance metrics

## Output Structure

**Data & Results:**
- `data/` - Input data
- `results/` - CV folds, predictions, metrics summaries

**Notebooks:**
- `notebooks/eda.ipynb` - Exploratory data analysis
- `notebooks/stacking_ablation.ipynb` - Stacking model breakdown and ablation study
- `notebooks/visualizations.ipynb` - Plots and statistical comparison (DeLong test)

**Figures:**
- `figures/` - All generated plots (ROC curves, PR curves, performance bars, stability, DeLong heatmap, CI bars)

## Key Files

- `models/` - Model training code (logistic, random forest, boosting, xgboost, stacking)
- `preprocessing/preprocess.py` - Data preparation
- `evaluation/metrics.py` - Metric computation

## Results Summary

Check `notebooks/visualizations.ipynb` for:
- ROC and precision-recall curves
- Cross-fold stability analysis
- Training time comparison
- Confidence intervals per metric
- Statistical significance (DeLong test)
