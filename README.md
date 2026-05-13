# Data Science Portfolio & Exercises

This repository is a collection of data science projects, exercises, and case studies. It is not intended to be a single unified project, but rather a space for documenting learning, experiments, and various analyses across different domains.

New cases and exercises are added over time as I explore different techniques and datasets.

## 🚀 Contents

Currently, the repository includes the following cases:

### 1. Smartphone Usage and Addiction Analysis
An end-to-end data science pipeline to analyze smartphone usage patterns and predict addiction levels.
- **Key Features:** Data cleaning, feature engineering (screen time ratios), modular preprocessing with `ColumnTransformer`.
- **Models:** Logistic Regression and Random Forest Classifier.
- **Evaluation:** RandomizedSearchCV for hyperparameter tuning, StratifiedKFold cross-validation, and Paired T-Tests for statistical model comparison.
- **Location:** `smartphone-addiction/`

## 🛠️ Tech Stack

The experiments in this repository are primarily built with:

- **Language:** Python 3.12+
- **Data Manipulation:** pandas, numpy
- **Machine Learning:** scikit-learn, xgboost
- **Visualization:** matplotlib, seaborn
- **Environment:** Jupyter Notebooks / IPykernel

## 📂 Repository Structure

```text
.
├── smartphone-addiction/       # Case study on smartphone usage
│   ├── data/                   # Dataset files
│   ├── documentation/          # Methodology and results (MD and TeX)
│   ├── plots/                  # Visualizations
│   └── *.ipynb                 # Analysis and experimentation notebooks
├── pyproject.toml              # Dependency management
└── README.md                   # This file
```

---
*This repository is continuously updated with new exercises and cases.*
