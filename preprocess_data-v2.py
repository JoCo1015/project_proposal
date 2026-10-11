import pandas as pd
import numpy as np
import re
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Preprocess-Data")

def clean_ram(ram_str) -> float:
    """Extract numeric RAM size in GB from text string."""
    if pd.isna(ram_str):
        return np.nan
    s = str(ram_str)
    match = re.search(r'(\d+)\s*GB', s, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match_num = re.search(r'\d+', s)
    return float(match_num.group(0)) if match_num else np.nan

def clean_storage(storage_str) -> float:
    """Extract total storage in GB from text string (handles TB and GB components)."""
    if pd.isna(storage_str):
        return np.nan
    s = str(storage_str).upper()
    total_gb = 0.0
    
    # Search Terabytes
    tb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*TB', s)
    for tb in tb_matches:
        total_gb += float(tb) * 1024.0
        
    # Search Gigabytes
    gb_matches = re.findall(r'(\d+)\s*GB', s)
    for gb in gb_matches:
        total_gb += float(gb)
        
    return float(total_gb) if total_gb > 0 else np.nan

def clean_price(price_str) -> float:
    """Extract numeric currency price value from formatted string."""
    if pd.isna(price_str):
        return np.nan
    cleaned = re.sub(r'[^\d.]', '', str(price_str))
    try:
        val = float(cleaned)
        return val if val > 0 else np.nan
    except ValueError:
        return np.nan

def extract_cpu_brand(cpu_str) -> str:
    """Classify CPU brand (Intel, AMD, Apple, Other)."""
    if pd.isna(cpu_str):
        return 'Other'
    s = str(cpu_str).upper()
    if any(k in s for k in ['INTEL', 'CORE', 'CELERON', 'PENTIUM', 'I3', 'I5', 'I7', 'I9']):
        return 'Intel'
    elif any(k in s for k in ['AMD', 'RYZEN', 'ATHLON']):
        return 'AMD'
    elif any(k in s for k in ['APPLE', 'M1', 'M2', 'M3', 'M4']):
        return 'Apple'
    return 'Other'

def extract_gpu_brand(gpu_str) -> str:
    """Classify GPU brand (NVIDIA, AMD, Intel, Apple, Integrated/Other)."""
    if pd.isna(gpu_str):
        return 'Integrated/Other'
    s = str(gpu_str).upper()
    if any(k in s for k in ['NVIDIA', 'GEFORCE', 'RTX', 'GTX', 'QUADRO']):
        return 'NVIDIA'
    elif any(k in s for k in ['AMD', 'RADEON']):
        return 'AMD'
    elif any(k in s for k in ['INTEL', 'UHD', 'IRIS', 'GRAPHICS']):
        return 'Intel'
    elif any(k in s for k in ['APPLE', 'M1', 'M2', 'M3', 'M4']):
        return 'Apple'
    return 'Integrated/Other'

def load_csv_with_fallback(file_path: str) -> pd.DataFrame:
    """Attempt reading CSV with encoding fallback (UTF-8, Latin-1, CP1252, ISO-8859-1)."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            logger.info(f"Attempting to load '{file_path}' with encoding='{enc}'...")
            df = pd.read_csv(file_path, encoding=enc)
            logger.info(f"Successfully loaded '{file_path}' using '{enc}' encoding.")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error reading file with encoding '{enc}': {e}")
            break
    raise ValueError(f"Could not read file '{file_path}' with any supported encoding.")

def preprocess_tabular_data(input_csv: str, output_csv: str):
    """Preprocess raw hardware dataset into normalized CSV format for model training."""
    logger.info(f"Reading raw dataset from: {input_csv}")
    df = load_csv_with_fallback(input_csv)
    logger.info(f"Initial raw dataset dimensions: {df.shape}")

    # Standardize column headers
    cols_lower = {col: col.lower().strip().replace(" ", "_") for col in df.columns}
    df = df.rename(columns=cols_lower)

    # Column Auto-detection
    ram_col = next((c for c in df.columns if 'ram' in c or 'memory' in c), None)
    storage_col = next((c for c in df.columns if 'storage' in c or 'disk' in c or 'ssd' in c or 'hdd' in c), None)
    price_col = next((c for c in df.columns if 'price' in c or 'cost' in c or 'msrp' in c), None)
    cpu_col = next((c for c in df.columns if 'cpu' in c or 'processor' in c), None)
    gpu_col = next((c for c in df.columns if 'gpu' in c or 'graphics' in c), None)

    logger.info(f"Detected columns - RAM: '{ram_col}', Storage: '{storage_col}', Price: '{price_col}', CPU: '{cpu_col}', GPU: '{gpu_col}'")

    # Feature Transformations
    if ram_col:
        df['ram_gb'] = df[ram_col].apply(clean_ram)
    if storage_col:
        df['storage_gb'] = df[storage_col].apply(clean_storage)
    if price_col:
        df['price_numeric'] = df[price_col].apply(clean_price)
    if cpu_col:
        df['cpu_brand'] = df[cpu_col].apply(extract_cpu_brand)
    if gpu_col:
        df['gpu_brand'] = df[gpu_col].apply(extract_gpu_brand)

    # Filter out missing essential values
    essential_cols = [c for c in ['price_numeric', 'ram_gb', 'storage_gb'] if c in df.columns]
    df_clean = df.dropna(subset=essential_cols).copy()

    logger.info(f"Cleaned dataset shape: {df_clean.shape}")
    df_clean.to_csv(output_csv, index=False)
    logger.info(f"Successfully saved clean dataset to: '{output_csv}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EdgeSpecs Data Preprocessing")
    parser.add_argument("--input", type=str, default="laptops.csv", help="Path to input raw CSV file")
    parser.add_argument("--output", type=str, default="cleaned_laptops.csv", help="Path to output clean CSV file")
    args = parser.parse_args()

    if os.path.exists(args.input):
        preprocess_tabular_data(args.input, args.output)
    else:
        logger.error(f"Input file '{args.input}' does not exist.")
