from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import onnxruntime as ort
import numpy as np
import re
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EdgeSpecs-FastAPI")

app = FastAPI(
    title="EdgeSpecs AI Valuation Engine",
    description="Edge AI REST API service for NLP hardware entity extraction and ONNX price valuation",
    version="1.0.0"
)

# Global ONNX Runtime Session Variable
ORT_SESSION = None
MODEL_PATH = "valuation_model.onnx"

@app.on_event("startup")
def load_onnx_model():
    global ORT_SESSION
    if os.path.exists(MODEL_PATH):
        try:
            ORT_SESSION = ort.InferenceSession(MODEL_PATH)
            logger.info(f"Successfully loaded ONNX valuation model from '{MODEL_PATH}' into ONNX Runtime.")
        except Exception as e:
            logger.warning(f"Failed to load ONNX model from '{MODEL_PATH}': {e}. Falling back to baseline engine.")
    else:
        logger.info(f"Model file '{MODEL_PATH}' not found. Using baseline heuristic estimation engine.")

# Request & Response Schemas
class SpecTextRequest(BaseModel):
    raw_text: str = Field(
        ..., 
        example="Selling PC with Intel i7-12700K, 16GB RAM, RTX 3060 GPU, and 1TB SSD",
        description="Unstructured seller text listing containing hardware specs."
    )

class ExtractedSpecs(BaseModel):
    ram_gb: float = Field(..., description="Extracted RAM capacity in Gigabytes.")
    storage_gb: float = Field(..., description="Extracted Storage capacity in Gigabytes.")
    cpu_brand: str = Field(..., description="Identified CPU manufacturer brand (Intel, AMD, Apple, Other).")
    gpu_brand: str = Field(..., description="Identified GPU manufacturer brand (NVIDIA, AMD, Intel, Apple, Other).")

class ValuationResponse(BaseModel):
    raw_text: str
    extracted_specs: ExtractedSpecs
    estimated_price_usd: float = Field(..., description="Predicted fair market price in USD.")
    price_range_min: float = Field(..., description="Lower estimated price bound (-12%).")
    price_range_max: float = Field(..., description="Upper estimated price bound (+12%).")
    performance_tier: str = Field(..., description="Assigned hardware performance tier category.")
    inference_engine: str = Field(..., description="Engine used for prediction (ONNX Runtime or Heuristic Baseline).")

# Rule-based NLP Entity Extraction Functions
def extract_ram(text: str) -> float:
    """Extract RAM size in GB from text string using regular expressions."""
    match = re.search(r'(\d+)\s*GB', text, re.IGNORECASE)
    return float(match.group(1)) if match else 8.0

def extract_storage(text: str) -> float:
    """Extract total storage capacity in GB from text string."""
    text_upper = text.upper()
    total_gb = 0.0
    
    # Check Terabyte entries
    tb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*TB', text_upper)
    for tb in tb_matches:
        total_gb += float(tb) * 1024.0
        
    # Check Gigabyte entries (filtering out standard RAM capacities to prevent false positives)
    gb_matches = re.findall(r'(\d+)\s*GB', text_upper)
    for gb in gb_matches:
        val = float(gb)
        if val not in [4.0, 8.0, 16.0, 32.0, 64.0]:
            total_gb += val
            
    return total_gb if total_gb > 0 else 512.0

def extract_cpu(text: str) -> str:
    """Classify CPU manufacturer brand from text string."""
    text_upper = text.upper()
    if any(k in text_upper for k in ['INTEL', 'CORE', 'I3', 'I5', 'I7', 'I9', 'CELERON', 'PENTIUM']):
        return 'Intel'
    elif any(k in text_upper for k in ['AMD', 'RYZEN', 'ATHLON']):
        return 'AMD'
    elif any(k in text_upper for k in ['APPLE', 'M1', 'M2', 'M3', 'M4']):
        return 'Apple'
    return 'Other'

def extract_gpu(text: str) -> str:
    """Classify GPU manufacturer brand from text string."""
    text_upper = text.upper()
    if any(k in text_upper for k in ['NVIDIA', 'GEFORCE', 'RTX', 'GTX', 'QUADRO']):
        return 'NVIDIA'
    elif any(k in text_upper for k in ['AMD', 'RADEON']):
        return 'AMD'
    elif any(k in text_upper for k in ['APPLE', 'M1', 'M2', 'M3', 'M4']):
        return 'Apple'
    return 'Intel'

def determine_tier(ram: float, price: float) -> str:
    """Assign performance tier category based on RAM and price valuation."""
    if price > 1800 or ram >= 32:
        return "High-End Workstation / Gaming"
    elif price > 900 or ram >= 16:
        return "Mid-Range / Business Workhorse"
    else:
        return "Budget / Everyday Computing"

@app.get("/", status_code=status.HTTP_200_OK)
def read_root():
    """Health check endpoint confirming API active status."""
    return {
        "service": "EdgeSpecs AI Valuation Engine",
        "status": "active",
        "onnx_model_loaded": ORT_SESSION is not None
    }

@app.post("/predict", response_model=ValuationResponse, status_code=status.HTTP_200_OK)
def predict_valuation(request: SpecTextRequest):
    """
    Main Endpoint: Parse raw text, extract entities, run ONNX/Heuristic valuation,
    and return structured JSON response.
    """
    text = request.raw_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text string cannot be empty.")

    # Step 1: Entity Extraction
    ram_gb = extract_ram(text)
    storage_gb = extract_storage(text)
    cpu_brand = extract_cpu(text)
    gpu_brand = extract_gpu(text)

    extracted_specs = ExtractedSpecs(
        ram_gb=ram_gb,
        storage_gb=storage_gb,
        cpu_brand=cpu_brand,
        gpu_brand=gpu_brand
    )

    engine_used = "Heuristic Baseline"
    estimated_price = 0.0

    # Step 2: Inference Execution (ONNX Runtime if model present, else Baseline Heuristic)
    if ORT_SESSION is not None:
        try:
            # Prepare inputs for ONNX model
            input_data = {
                'ram_gb': np.array([[ram_gb]], dtype=np.float32),
                'storage_gb': np.array([[storage_gb]], dtype=np.float32),
                'cpu_brand': np.array([[cpu_brand]], dtype=object),
                'gpu_brand': np.array([[gpu_brand]], dtype=object)
            }
            outputs = ORT_SESSION.run(None, input_data)
            estimated_price = round(float(outputs[0][0]), 2)
            engine_used = "ONNX Runtime Engine"
        except Exception as e:
            logger.warning(f"ONNX execution error: {e}. Falling back to baseline formula.")

    if engine_used == "Heuristic Baseline":
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
        extracted_specs=extracted_specs,
        estimated_price_usd=estimated_price,
        price_range_min=price_min,
        price_range_max=price_max,
        performance_tier=tier,
        inference_engine=engine_used
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
