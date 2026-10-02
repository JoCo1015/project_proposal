import pandas as pd
import numpy as np
import re
import os
import argparse

def clean_ram(ram_str):
    """Extract numeric RAM size in GB from string (e.g., '16GB', '8 GB DDR4')."""
    if pd.isna(ram_str):
        return np.nan
    match = re.search(r'(\d+)\s*GB', str(ram_str), re.IGNORECASE)
    if match:
        return int(match.group(1))
    match_num = re.search(r'\d+', str(ram_str))
    return int(match_num.group(0)) if match_num else np.nan

def clean_storage(storage_str):
    """Extract total storage in GB (e.g., '512GB SSD', '1TB HDD', '256GB SSD + 1TB HDD')."""
    if pd.isna(storage_str):
        return np.nan
    s = str(storage_str).upper()
    total_gb = 0.0
    
    # Search for TB
    tb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*TB', s)
    for tb in tb_matches:
        total_gb += float(tb) * 1024.0
        
    # Search for GB
    gb_matches = re.findall(r'(\d+)\s*GB', s)
    for gb in gb_matches:
        total_gb += float(gb)
        
    return int(total_gb) if total_gb > 0 else np.nan

def clean_price(price_str):
    """Extract numeric price value from currency strings."""
    if pd.isna(price_str):
        return np.nan
    cleaned = re.sub(r'[^\d.]', '', str(price_str))
    try:
        return float(cleaned)
    except ValueError:
        return np.nan

def extract_cpu_brand(cpu_str):
    """Extract CPU brand (Intel, AMD, Apple, Other)."""
    if pd.isna(cpu_str):
        return 'Other'
    s = str(cpu_str).upper()
    if any(k in s for k in ['INTEL', 'CORE', 'CELERON', 'PENTIUM', 'I3', 'I5', 'I7', 'I9']):
        return 'Intel'
    elif any(k in s for k in ['AMD', 'RYZEN', 'ATHLON']):
        return 'AMD'
    elif any(k in s for k in ['APPLE', 'M1', 'M2', 'M3']):
        return 'Apple'
    return 'Other'

def extract_gpu_brand(gpu_str):
    """Extract GPU brand (NVIDIA, AMD, Intel, Apple, Integrated/Other)."""
    if pd.isna(gpu_str):
        return 'Integrated/Other'
    s = str(gpu_str).upper()
    if any(k in s for k in ['NVIDIA', 'GEFORCE', 'RTX', 'GTX', 'QUADRO']):
        return 'NVIDIA'
    elif any(k in s for k in ['AMD', 'RADEON']):
        return 'AMD'
    elif any(k in s for k in ['INTEL', 'UHD', 'IRIS', 'GRAPHICS']):
        return 'Intel'
    elif any(k in s for k in ['APPLE', 'M1', 'M2', 'M3']):
        return 'Apple'
    return 'Integrated/Other'

def read_csv_with_encoding_fallback(filepath):
    """Read CSV file trying UTF-8, Latin-1, and CP1252 encodings."""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"Successfully read CSV with encoding: '{enc}'")
            return df
        except UnicodeDecodeError:
            continue
    # Fallback with replacement
    print("Warning: Standard encodings failed. Reading with encoding_errors='replace'")
    return pd.read_csv(filepath, encoding='utf-8', encoding_errors='replace')

def preprocess_tabular_data(input_csv, output_csv):
    """Clean tabular hardware dataset for regression training."""
    print(f"Reading raw dataset from: {input_csv}")
    df = read_csv_with_encoding_fallback(input_csv)
    print(f"Initial shape: {df.shape}")

    # Column name mapping
    cols_lower = {col: col.lower().strip() for col in df.columns}
    df = df.rename(columns=cols_lower)

    # Detect key target and feature columns
    ram_col = next((c for c in df.columns if 'ram' in c or 'memory' in c), None)
    storage_col = next((c for c in df.columns if 'storage' in c or 'disk' in c or 'ssd' in c or 'hdd' in c), None)
    price_col = next((c for c in df.columns if 'price' in c or 'cost' in c or 'msrp' in c or 'euros' in c or 'eur' in c), None)
    cpu_col = next((c for c in df.columns if 'cpu' in c or 'processor' in c), None)
    gpu_col = next((c for c in df.columns if 'gpu' in c or 'graphics' in c), None)

    print(f"Detected columns - RAM: {ram_col}, Storage: {storage_col}, Price: {price_col}, CPU: {cpu_col}, GPU: {gpu_col}")

    # Clean features
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

    # Drop missing target / essential features
    essential_cols = [c for c in ['price_numeric', 'ram_gb', 'storage_gb'] if c in df.columns]
    df_clean = df.dropna(subset=essential_cols).copy()

    print(f"Cleaned dataset shape: {df_clean.shape}")
    df_clean.to_csv(output_csv, index=False)
    print(f"Successfully saved clean dataset to: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess PC Hardware Datasets for ML Valuation Engine")
    parser.add_argument("--input", type=str, default="laptops.csv", help="Path to input raw CSV file")
    parser.add_argument("--output", type=str, default="cleaned_hardware_data.csv", help="Path to output clean CSV file")
    args = parser.parse_args()

    if os.path.exists(args.input):
        preprocess_tabular_data(args.input, args.output)
    else:
        print(f"Usage example: python preprocess_data.py --input path/to/laptops.csv --output cleaned_hardware.csv")
