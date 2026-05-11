# Results and Discussion

## 1. Model Performance Results
The predictive performance of the Logistic Regression and Random Forest models was evaluated using several metrics on a 20% hold-out test set. The optimized Random Forest model demonstrated superior performance across all primary metrics.

| Metric | Logistic Regression | Random Forest |
| :--- | :---: | :---: |
| Accuracy | ~0.932 | 0.934 |
| Precision | ~0.958 | 0.963 |
| Recall | ~0.938 | 0.942 |
| F1-score | ~0.941 | 0.952 |
| ROC-AUC | ~0.985 | 0.988 |

### 1.1 Statistical Significance
A paired t-test was performed comparing the F1-scores of both models over 10-fold cross-validation. The test yielded a p-value of 0.0134, allowing us to reject the null hypothesis at a 5% significance level. This confirms that the Random Forest model's improvement over Logistic Regression is statistically significant (T-Stat: -3.066).

## 2. Feature Importance Analysis
The Random Forest model's feature importance analysis identified the most influential behavioral predictors of smartphone addiction. The results highlight a heavy reliance on usage duration and psychological state.

- **Daily Screen Time Hours (53.0%):** By far the most significant predictor, indicating a direct relationship between total device usage and addiction risk.
- **Social Media Hours (24.4%):** The second most critical factor, suggesting that the *type* of content (specifically social networking) is a major driver.
- **Stress Level (8.5%):** Indicates a correlation between higher reported stress and addictive usage patterns.
- **Other Factors:** Sleep hours, work/study hours, and notification frequency each contributed approximately 2% to the model's predictive power.

## 3. Discussion
The findings suggest that smartphone addiction is primarily driven by the quantity of time spent on the device, particularly on social media platforms. The high ROC-AUC (0.988) indicates that the Random Forest model is exceptionally capable of distinguishing between "Addicted" and "Not Addicted" individuals.

The statistically significant superiority of the Random Forest model suggests that non-linear interactions between variables—such as the interplay between stress levels and screen time—play a crucial role in predicting addiction. While linear models like Logistic Regression perform well, they fail to capture the complexity of behavioral patterns as effectively as ensemble-based methods.

Furthermore, the relatively low importance of the engineered `screen_time_to_social_network_ratio` (0.3%) compared to the raw `social_media_hours` suggests that the absolute volume of social media usage is a more direct indicator of addiction than its proportion relative to other activities.
