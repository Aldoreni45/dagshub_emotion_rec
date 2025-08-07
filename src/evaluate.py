import os
import pandas as pd
from sklearn.metrics import classification_report
import joblib

def main():
    features_path = os.path.join("data", "features", "features.csv")
    model_path = os.path.join("models", "model.pkl")
    df = pd.read_csv(features_path)
    X = df.drop("label", axis=1)
    y = df["label"]
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    main()