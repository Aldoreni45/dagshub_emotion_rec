import os
import pandas as pd

def load_features_csv(csv_path):
    return pd.read_csv(csv_path)

def save_processed_data(df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    raw_csv = os.path.join("ravdess_features.csv")
    processed_path = os.path.join("data", "processed", "processed_features.csv")
    df = load_features_csv(raw_csv)
    save_processed_data(df, processed_path)
    print(f"Processed data saved to {processed_path}")