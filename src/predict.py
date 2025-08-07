import os
import pandas as pd
import joblib

def predict(input_csv, model_path="models/model.pkl"):
    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    if "label" in df.columns:
        df = df.drop("label", axis=1)
    preds = model.predict(df)
    return preds

if __name__ == "__main__":
    input_csv = os.path.join("data", "features", "features.csv")
    preds = predict(input_csv)
    print("Predictions:", preds)