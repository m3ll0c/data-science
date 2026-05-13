# Case Study: Smartphone Usage and Addiction Analysis

This case study focuses on analyzing behavioral usage patterns to predict smartphone addiction levels. Using a dataset of 7,500 observations, we implement a full data science pipeline—from cleaning and feature engineering to statistical model validation.

## 📊 Key Findings

- **Top Predictors:** Total daily screen time (53%) and social media usage (24.4%) are the strongest indicators of addiction.
- **Psychological Link:** Stress levels are the most significant non-behavioral predictor (8.5%).
- **Best Model:** The **Random Forest Classifier** significantly outperformed Logistic Regression, achieving an **F1-score of 0.952** and an **ROC-AUC of 0.988**.

## 🚀 Methodology

The analysis follows a modular predictive modeling lifecycle:

1.  **Data Preparation:** 
    - Removal of non-informative identifiers (User_ID, Transaction_ID).
    - Consistency checks between social media hours and total screen time.
2.  **Feature Engineering:** 
    - Created `screen_time_to_social_network_ratio` to measure social media intensity.
3.  **Preprocessing:** 
    - Robust pipeline using `ColumnTransformer`.
    - Standardization for numerical features and One-Hot Encoding for categorical variables.
4.  **Modeling & Tuning:** 
    - Comparison between **Logistic Regression** (baseline) and **Random Forest**.
    - Hyperparameter optimization via `RandomizedSearchCV` with 5-fold cross-validation.
5.  **Statistical Validation:** 
    - **Paired T-Test** on 10-fold cross-validation results to confirm the statistical significance of model improvements.

## 📈 Results Summary

| Metric | Logistic Regression | Random Forest |
| :--- | :---: | :---: |
| **Accuracy** | 0.932 | **0.934** |
| **F1-score** | 0.941 | **0.952** |
| **ROC-AUC** | 0.985 | **0.988** |

*A paired t-test confirmed the Random Forest's superiority with a p-value of 0.0134.*

## 📂 Folder Structure

- `data/`: Contains the raw dataset (`Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`).
- `documentation/`: Detailed reports in Markdown and LaTeX formats (Methodology, Results, Conclusion).
- `plots/`: Visualizations of feature importance, confusion matrices, and ROC curves.
- `PEL309-Full_Pipeline.ipynb`: The main notebook containing the end-to-end execution.
- `PEL309-Paper_Experiments.ipynb`: Detailed experimental trials and tuning.
- `PEL309-Paper_Visualization.ipynb`: Specialized notebook for generating high-quality plots.
- `pipeline.py`: A modular Python script for automated execution of the entire classification pipeline.

## 🚀 Usage

### Running the Notebooks
Open the `.ipynb` files in a Jupyter environment to explore the analysis step-by-step.

### Running the Standalone Pipeline
To run the end-to-end pipeline (cleaning, engineering, training, and evaluation) in a single command:

```bash
uv run pipeline.py
```
*Note: Make sure you are in the `smartphone-addiction/` directory or have the root dependencies installed.*

## 🛠️ Requirements

Dependencies are managed at the root level. Ensure you have the environment set up as described in the [root README](../README.md).

---
*Developed as part of the PEL309 Data Science curriculum.*
