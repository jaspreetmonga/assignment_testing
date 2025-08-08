import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def split_and_preprocess(input_path, output_path, test_size, random_state, target_column):
    # Step 1: Load dataset
    df = pd.read_csv(input_path)
    print(f"Loaded dataset from {input_path} with shape {df.shape}")

    # Step 2: Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Fill missing values only for numeric columns using median
    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # Step 3: Handle categorical features (one-hot encoding)
    categorical_cols = ['ocean_proximity']  # Adjust if more categorical columns exist
    X_encoded = pd.get_dummies(X, columns=categorical_cols)
    print(f"Applied one-hot encoding. New feature shape: {X_encoded.shape}")

    # Step 4: Train-test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state
    )
    print(f"Split into: X_train {X_train_raw.shape}, X_test {X_test_raw.shape}, "
          f"y_train {y_train.shape}, y_test {y_test.shape}")

    # Step 5: Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Step 6: Create directories for saving outputs
    raw_dir = os.path.join(output_path, "raw")
    print("raw_dir value: ", raw_dir)
    processed_dir = os.path.join(output_path, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Step 7: Save raw (unscaled) data
    X_train_raw.to_csv(os.path.join(raw_dir, "X_train_raw.csv"), index=False)
    X_test_raw.to_csv(os.path.join(raw_dir, "X_test_raw.csv"), index=False)
    y_train.to_csv(os.path.join(raw_dir, "y_train_raw.csv"), index=False)
    y_test.to_csv(os.path.join(raw_dir, "y_test_raw.csv"), index=False)

    # Step 8: Save processed (scaled) data as DataFrame for consistency
    pd.DataFrame(X_train_scaled, columns=X_train_raw.columns).to_csv(
        os.path.join(processed_dir, "X_train.csv"), index=False
    )
    pd.DataFrame(X_test_scaled, columns=X_test_raw.columns).to_csv(
        os.path.join(processed_dir, "X_test.csv"), index=False
    )
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    # Step 9: Save scaler
    scaler_path = os.path.join("models", "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    print(f"   Data preprocessing complete.\n"
          f"   Raw data saved in {raw_dir}\n"
          f"   Processed data saved in {processed_dir}\n"
          f"   Scaler saved at {scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and preprocess dataset.")
    parser.add_argument("--in", dest="input_path", required=True, help="Path to raw data CSV")
    parser.add_argument("--out", dest="output_path", required=True, help="Directory to save processed data")
    parser.add_argument("--test_size", type=float, required=True, help="Test set size fraction")
    parser.add_argument("--random_state", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--target", dest="target_column", required=True, help="Name of target column")

    args = parser.parse_args()

    split_and_preprocess(
        input_path=args.input_path,
        output_path=args.output_path,
        test_size=args.test_size,
        random_state=args.random_state,
        target_column=args.target_column
    )
