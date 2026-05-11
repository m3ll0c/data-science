# Methodology

This section describes the data science pipeline implemented to analyze smartphone usage patterns and predict addiction levels. The workflow follows a standard predictive modeling lifecycle, including data cleaning, feature engineering, transformation, model selection, and statistical validation.

## 1. Data Description and Preparation
The dataset comprises 7,500 observations of individual smartphone usage behaviors. Each record includes demographic information, usage metrics (hours spent on various activities), and psychological impact indicators.

### 1.1 Data Cleaning
To ensure data integrity, the following steps were performed:
- **Identifier Removal:** Non-informative columns such as `User_ID` and `Transaction_ID` were excluded from the analysis.
- **Consistency Verification:** Records were filtered to ensure logical consistency, specifically verifying that `Social_Media_Hours` does not exceed the total `Daily_Screen_Time_Hours`.

### 1.2 Feature Engineering
A new feature, `screen_time_to_social_network_ratio`, was calculated to represent the proportion of total screen time dedicated specifically to social networking. This feature aims to capture the intensity of social media usage relative to overall device interaction.

## 2. Feature Transformation
A modular preprocessing pipeline was constructed using scikit-learn's `ColumnTransformer` to handle different data types effectively:

- **Numerical Features:** Features including `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, and the engineered ratio were standardized using `StandardScaler` to have a mean of zero and unit variance.
- **Categorical Features:** Qualitative variables such as `stress_level` and `academic_work_impact` were transformed using `OneHotEncoder`. Binary variables were encoded using a `drop='if_binary'` strategy to minimize redundancy.

## 3. Predictive Modeling
Two distinct classification algorithms were evaluated for their ability to predict smartphone addiction:

1.  **Logistic Regression:** Serving as a baseline linear model.
2.  **Random Forest Classifier:** An ensemble method utilizing decision trees to capture non-linear relationships and feature interactions.

### 3.1 Hyperparameter Optimization
Both models underwent hyperparameter tuning using `RandomizedSearchCV`. The optimization process utilized a 5-fold `StratifiedKFold` cross-validation strategy to maximize the F1-score, ensuring a balanced performance between precision and recall.

## 4. Model Evaluation and Validation
The performance of the models was assessed on a dedicated test set (20% of the total data) using the following metrics:
- Accuracy
- Precision and Recall
- F1-score (Primary metric)
- ROC-AUC

### 4.1 Statistical Significance Testing
To rigorously compare the models, a **Paired T-Test** was conducted on the results of a 10-fold cross-validation. This test evaluated the null hypothesis that there is no significant difference in the F1-scores of the Logistic Regression and Random Forest models.

### 4.2 Feature Importance
For the Random Forest model, feature importance was calculated based on Gini impurity reduction to identify the primary behavioral drivers of smartphone addiction.
