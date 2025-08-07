import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def build_features(df):
    # Example: Standardize all features except the label
    feature_cols = [col for col in df.columns if col != "label"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

if __name__ == "__main__":
    processed_path = os.path.join("data", "processed", "processed_features.csv")
    features_path = os.path.join("data", "features", "features.csv")
    df = pd.read_csv(processed_path)
    df = build_features(df)
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    df.to_csv(features_path, index=False)
    print(f"Features saved to {features_path}")