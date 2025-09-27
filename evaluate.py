import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.base import is_classifier, is_regressor

def evaluate(model_path, input_path, output_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Run train.py first.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(input_path).dropna(subset=['rating_number', 'rating_text'])
    X = df.drop(columns=['rating_number', 'rating_text'])

    estimator = model.named_steps["model"]

    if is_regressor(estimator):
        y = df['rating_number'].astype(float)  # ensure numeric
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print("\nRegression Evaluation:")
        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    elif is_classifier(estimator):
        y = (
            df['rating_text']
            .replace({
                "Poor": 0, "Average": 0,
                "Good": 1, "Very Good": 1, "Excellent": 1
            })
            .astype(int)   # explicit cast
        )
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        print("\nClassification Evaluation:")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    else:
        print("Error: Could not determine model type.")
        return

    df['predicted'] = y_pred
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

if __name__ == "__main__":
    model_path = "models/model.pkl"
    input_path = "data/features.csv"
    output_path = "results/predictions.csv"
    evaluate(model_path, input_path, output_path)
