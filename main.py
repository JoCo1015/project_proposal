from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import re
import os

app = FastAPI(
    title="EdgeSpecs AI Valuation Engine",
    description="Edge AI REST API service for NLP spec extraction and ONNX price valuation",
    version="1.0.0"
)

# Request & Response Schemas
class SpecTextRequest(BaseModel):
    raw_text: str

class ExtractedSpecs(BaseModel):
    ram_gb: float
    storage_gb: float
    cpu_brand: str
    gpu_brand: str

class ValuationResponse(BaseModel):
    raw_text: str
    extracted_specs: ExtractedSpecs
    estimated_price_usd: float
    price_range_min: float
    price_range_max: float
    performance_tier: str

# Helper functions for rule-based extraction fallback
def extract_ram(text: str) -> float:
    match = re.search(r'(\d+)\s*GB', text, re.IGNORECASE)
    return float(match.group(1)) if match else 8.0

def extract_storage(text: str) -> float:
    text_upper = text.upper()
    total_gb = 0.0
    tb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*TB', text_upper)
    for tb in tb_matches:
        total_gb += float(tb) * 1024.0
    gb_matches = re.findall(r'(\d+)\s*GB', text_upper)
    for gb in gb_matches:
        if float(gb) not in [8.0, 16.0, 32.0, 64.0]:  # Avoid RAM false positives
            total_gb += float(gb)
    return total_gb if total_gb > 0 else 512.0

def extract_cpu(text: str) -> str:
    text_upper = text.upper()
    if any(k in text_upper for k in ['INTEL', 'CORE', 'I3', 'I5', 'I7', 'I9']):
        return 'Intel'
    elif any(k in text_upper for k in ['AMD', 'RYZEN']):
        return 'AMD'
    elif any(k in text_upper for k in ['APPLE', 'M1', 'M2', 'M3']):
        return 'Apple'
    return 'Other'

def extract_gpu(text: str) -> str:
    text_upper = text.upper()
    if any(k in text_upper for k in ['NVIDIA', 'GEFORCE', 'RTX', 'GTX']):
        return 'NVIDIA'
    elif any(k in text_upper for k in ['AMD', 'RADEON']):
        return 'AMD'
    elif any(k in text_upper for k in ['APPLE', 'M1', 'M2', 'M3']):
        return 'Apple'
    return 'Intel'

def determine_tier(ram: float, price: float) -> str:
    if price > 1800 or ram >= 32:
        return "High-End Workstation / Gaming"
    elif price > 900 or ram >= 16:
        return "Mid-Range / Business Workhorse"
    else:
        return "Budget / Everyday Computing"

@app.get("/")
def read_root():
    return {"message": "EdgeSpecs AI Service is Running", "status": "active"}

@app.post("/predict", response_model=ValuationResponse)
def predict_valuation(request: SpecTextRequest):
    text = request.raw_text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Step 1: Extract Entities
    ram_gb = extract_ram(text)
    storage_gb = extract_storage(text)
    cpu_brand = extract_cpu(text)
    gpu_brand = extract_gpu(text)

    specs = ExtractedSpecs(
        ram_gb=ram_gb,
        storage_gb=storage_gb,
        cpu_brand=cpu_brand,
        gpu_brand=gpu_brand
    )

    # Step 2: Heuristic / ONNX Inference
    # Baseline estimation logic for demo/fallback
    base_price = (ram_gb * 25.0) + (storage_gb * 0.4)
    if cpu_brand == 'Intel': base_price += 350
    elif cpu_brand == 'AMD': base_price += 320
    elif cpu_brand == 'Apple': base_price += 500

    if gpu_brand == 'NVIDIA': base_price += 450
    elif gpu_brand == 'AMD': base_price += 300

    estimated_price = round(base_price, 2)
    price_min = round(estimated_price * 0.88, 2)
    price_max = round(estimated_price * 1.12, 2)
    tier = determine_tier(ram_gb, estimated_price)

    return ValuationResponse(
        raw_text=text,
        extracted_specs=specs,
        estimated_price_usd=estimated_price,
        price_range_min=price_min,
        price_range_max=price_max,
        performance_tier=tier
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
