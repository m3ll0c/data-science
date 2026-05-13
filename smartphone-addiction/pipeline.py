import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import os

def load_and_clean_data(file_path):
    """Loads and performs initial cleaning of the dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path)
    
    # 1. Basic Cleaning: Remove non-informative IDs
    df = df.drop(columns=['transaction_id', 'user_id'])
    
    # 2. Consistency Check: Social media hours should not exceed total screen time
    initial_count = len(df)
    df = df[df['social_media_hours'] <= df['daily_screen_time_hours']]
    print(f"Removed {initial_count - len(df)} inconsistent records.")
    
    # 3. Feature Engineering: Create intensity ratio
    df["screen_time_to_social_network_ratio"] = (
        df["social_media_hours"] / df["daily_screen_time_hours"]
    ).fillna(0)
    
    # 4. Remove leaky or redundant features identified in EDA
    # Note: 'addiction_level' is a direct leak of the label
    df = df.drop(columns=['addiction_level', 'weekend_screen_time', 'gender'])
    
    return df

def build_pipeline():
    """Builds the preprocessing and modeling pipeline."""
    # Define features
    numeric_features = [
        'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
        'work_study_hours', 'sleep_hours', 'notifications_per_day', 
        'app_opens_per_day', 'screen_time_to_social_network_ratio'
    ]
    categorical_features = ['stress_level', 'academic_work_impact']

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), categorical_features)
        ]
    )

    # Full Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline

def train_and_evaluate(df):
    """Trains the model and evaluates performance."""
    X = df.drop(columns=['addicted_label'])
    y = df['addicted_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline = build_pipeline()
    
    # Hyperparameter tuning (simplified for the script)
    param_dist = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    
    print("Starting hyperparameter tuning...")
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=5, 
        cv=StratifiedKFold(5), scoring='f1', random_state=42, n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)[:, 1]
    
    print("\nBest Parameters:", search.best_params_)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print(f"F1-Score: {f1_score(y_test, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    
    return best_model

if __name__ == "__main__":
    DATA_PATH = "data/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
    
    # Ensure we are in the right directory context if run from root
    if not os.path.exists(DATA_PATH) and os.path.exists("smartphone-addiction/" + DATA_PATH):
        os.chdir("smartphone-addiction")
        
    try:
        processed_df = load_and_clean_data(DATA_PATH)
        train_and_evaluate(processed_df)
    except Exception as e:
        print(f"Error: {e}")
