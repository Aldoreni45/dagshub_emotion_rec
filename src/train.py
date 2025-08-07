import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.model import get_model
import joblib

def main():
    features_path = os.path.join("data", "features", "features.csv")
    model_path = os.path.join("models", "model.pkl")
    df = pd.read_csv(features_path)
    X = df.drop("label", axis=1)
    y = df["label"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = get_model()
    model.fit(X_train, y_train)
    acc = model.score(X_val, y_val)
    print(f"Validation accuracy: {acc:.4f}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()