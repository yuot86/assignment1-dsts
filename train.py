import pandas as pd
import pickle
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def train_and_compare(input_path, reg_results_csv, cls_results_csv, model_out, metrics_out):
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['rating_number', 'rating_text'])

    # Features and targets
    X = df.drop(columns=['rating_number', 'rating_text'])
    y_reg = df['rating_number']
    y_cls = df['rating_text'].replace({
        "Poor": 0, "Average": 0,
        "Good": 1, "Very Good": 1, "Excellent": 1
    })

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(exclude=['int64', 'float64']).columns

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # ---------------- Regression ---------------- #
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    reg_results = []
    best_model = None
    best_r2 = -1

    for name, model in reg_models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
        pipe.fit(X_train_r, y_train_r)
        y_pred = pipe.predict(X_test_r)
        mse = mean_squared_error(y_test_r, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_r, y_pred)
        r2 = r2_score(y_test_r, y_pred)

        reg_results.append({"Model": name, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2})

        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipe

        # Save each model separately
        os.makedirs("models", exist_ok=True)
        with open(f"models/{name.replace(' ', '_').lower()}.pkl", "wb") as f:
            pickle.dump(pipe, f)

    # Save best regression model as canonical model.pkl for DVC
    if best_model:
        with open(model_out, "wb") as f:
            pickle.dump(best_model, f)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(reg_results).to_csv(reg_results_csv, index=False)

    # ---------------- Classification ---------------- #
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_cls, test_size=0.2, random_state=42
    )
    cls_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    cls_results = []
    for name, model in cls_models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
        pipe.fit(X_train_c, y_train_c)
        y_pred = pipe.predict(X_test_c)
        acc = accuracy_score(y_test_c, y_pred)
        prec = precision_score(y_test_c, y_pred)
        rec = recall_score(y_test_c, y_pred)
        f1 = f1_score(y_test_c, y_pred)
        cls_results.append({"Model": name, "Accuracy": acc,
                            "Precision": prec, "Recall": rec, "F1": f1})
        with open(f"models/{name.replace(' ', '_').lower()}.pkl", "wb") as f:
            pickle.dump(pipe, f)

    pd.DataFrame(cls_results).to_csv(cls_results_csv, index=False)

    # Save a summary metrics.json for DVC
    summary = {
        "Best Regression Model": [m for m in reg_results if m["R2"] == best_r2][0],
        "Classification Models": cls_results
    }
    with open(metrics_out, "w") as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    input_path = "data/features.csv"
    reg_results_csv = "results/regression_metrics.csv"
    cls_results_csv = "results/classification_metrics.csv"
    model_out = "models/model.pkl"            # DVC-required output
    metrics_out = "results/metrics.json"      # DVC metrics file
    train_and_compare(input_path, reg_results_csv, cls_results_csv, model_out, metrics_out)
