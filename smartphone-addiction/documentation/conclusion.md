# Conclusion

This study successfully implemented and evaluated a data science pipeline to predict smartphone addiction based on behavioral usage patterns. By analyzing a dataset of 7,500 individuals, we identified key predictors and compared the efficacy of linear and non-linear classification models.

The primary findings of this research are:
1.  **Model Performance:** The Random Forest Classifier emerged as the superior model, achieving an F1-score of 0.952 and an ROC-AUC of 0.988. Statistical validation via a paired t-test ($p = 0.0134$) confirmed that the ensemble method significantly outperformed the baseline Logistic Regression model.
2.  **Key Behavioral Drivers:** Smartphone addiction is most strongly predicted by the total volume of usage. `Daily Screen Time Hours` (53.0%) and `Social Media Hours` (24.4%) accounted for over three-quarters of the model's predictive importance.
3.  **Psychological Factors:** Reported `Stress Level` was the most significant non-usage-based predictor (8.5%), highlighting the intersection between psychological well-being and device interaction patterns.

In conclusion, while total screen time is the most immediate indicator of addiction risk, the specific focus on social media platforms is a critical contributing factor. The high predictive accuracy of the Random Forest model suggests that behavioral data can effectively identify individuals at risk of smartphone addiction, enabling the development of targeted digital well-being interventions.

Future work should focus on longitudinal studies to determine the causal relationships between these behavioral drivers and addiction levels over time. Additionally, exploring more granular app-category data could further refine the identification of problematic usage patterns.
