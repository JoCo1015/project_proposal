import pandas as pd
import numpy as np
import argparse
import os
import logging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Train-ONNX")

def train_and_export_model(cleaned_csv_path: str, model_output_path: str):
    """
    Train a Random Forest Regressor on PC hardware specs and convert to ONNX binary format.
    """
    if not os.path.exists(cleaned_csv_path):
        logger.error(f"Input file '{cleaned_csv_path}' not found. Please run preprocessing first.")
        return

    logger.info(f"Loading cleaned dataset from: {cleaned_csv_path}")
    df = pd.read_csv(cleaned_csv_path)

    # Feature selection
    feature_cols = ['ram_gb', 'storage_gb', 'cpu_brand', 'gpu_brand']
    target_col = 'price_numeric'

    available_features = [col for col in feature_cols if col in df.columns]
    
    if target_col not in df.columns or len(available_features) < 2:
        logger.error("Dataset missing required target or feature columns.")
        return

    X = df[available_features].copy()
    y = df[target_col].copy()

    numeric_features = [col for col in ['ram_gb', 'storage_gb'] if col in X.columns]
    categorical_features = [col for col in ['cpu_brand', 'gpu_brand'] if col in X.columns]

    # Preprocessing Pipeline
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

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Training Random Forest Regressor Model...")
    model_pipeline.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info("--- Model Evaluation Metrics ---")
    logger.info(f"Mean Absolute Error (MAE): ${mae:.2f} USD")
    logger.info(f"R^2 Score (Coefficient of Determination): {r2:.4f}")

    # ONNX Conversion using skl2onnx
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType

        logger.info("Converting scikit-learn pipeline to ONNX binary format...")
        
        initial_types = []
        for col in available_features:
            if col in numeric_features:
                initial_types.append((col, FloatTensorType([None, 1])))
            else:
                initial_types.append((col, StringTensorType([None, 1])))

        onnx_model = convert_sklearn(model_pipeline, initial_types=initial_types)

        with open(model_output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"Successfully exported ONNX model binary to: '{model_output_path}'")
    except ImportError:
        logger.error("Package 'skl2onnx' is not installed. Please install via: pip install skl2onnx")
    except Exception as e:
        logger.error(f"An error occurred during ONNX model conversion: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EdgeSpecs - Train Valuation Regressor & Export ONNX")
    parser.add_argument("--input", type=str, default="cleaned_laptops.csv", help="Path to cleaned hardware dataset CSV")
    parser.add_argument("--output", type=str, default="valuation_model.onnx", help="Output path for ONNX model binary")
    args = parser.parse_args()

    train_and_export_model(args.input, args.output)
