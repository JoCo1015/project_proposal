import pandas as pd
import numpy as np
import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_export_model(cleaned_csv_path, model_output_path):
    print(f"Loading cleaned dataset from: {cleaned_csv_path}")
    df = pd.read_csv(cleaned_csv_path)

    # Feature selection
    feature_cols = ['ram_gb', 'storage_gb', 'cpu_brand', 'gpu_brand']
    target_col = 'price_numeric'

    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features].copy()
    y = df[target_col].copy()

    numeric_features = [col for col in ['ram_gb', 'storage_gb'] if col in X.columns]
    categorical_features = [col for col in ['cpu_brand', 'gpu_brand'] if col in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit model
    print("Training Valuation Regressor Model...")
    model_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Model Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"R^2 Score: {r2:.4f}")

    # Convert to ONNX
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType

        print("\nConverting trained pipeline to ONNX format...")
        
        # Define types for skl2onnx
        initial_type = []
        for col in available_features:
            if col in numeric_features:
                initial_type.append((col, FloatTensorType([None, 1])))
            else:
                initial_type.append((col, StringTensorType([None, 1])))

        onnx_model = convert_sklearn(model_pipeline, initial_types=initial_type)

        with open(model_output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"Successfully saved ONNX model to: {model_output_path}")
    except ImportError:
        print("\n[Warning] skl2onnx not installed. Please run: pip install skl2onnx")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Valuation Model & Convert to ONNX")
    parser.add_argument("--input", type=str, default="cleaned_laptops.csv", help="Path to cleaned dataset CSV")
    parser.add_argument("--output", type=str, default="valuation_model.onnx", help="Path for exported ONNX model")
    args = parser.parse_args()

    if os.path.exists(args.input):
        train_and_export_model(args.input, args.output)
    else:
        print(f"Error: Path '{args.input}' does not exist.")
