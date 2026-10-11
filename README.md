# EdgeSpecs: Laptop AI Valuation Engine 💻⚡

**EdgeSpecs** is a low-latency Edge AI application that extracts laptop specifications (`CPU`, `RAM`, `GPU`, `Storage`) from text listings and predicts fair market values using a quantized ONNX model on FastAPI.

---

## 📂 Project Structure

```text
edgespecs-ai/
├── datasets/                  # Place raw laptops.csv here
│   └── laptops.csv
├── preprocess_data-v2.py      # Cleans raw dataset
├── prepare_ner_dataset.py     # Prepares spaCy NLP training data
├── train_valuation_onnx.py    # Trains ML model & exports valuation_model.onnx
├── main.py                    # FastAPI server
├── client.py                  # CLI REST client
├── requirements.txt           # Python dependencies
├── setup-environment.sh       # Auto setup script
└── README.md                  # Documentation
```

---

# Dependencies & Requirements📃
The dependencies such as ONNX and Torchvision are shown in the `requirements.txt` file. Aside from that an application that can run python programs should be included whether its a source code editor like VSC or a notebook based browser like Jupyter and Colab (though this needs a bit of code tweaking so the former is better to use even if the latter works).

* **Python Version:** `Python 3.10+` (Tested on Python 3.12)
* **Backend API Framework:** `FastAPI` + `Uvicorn`
* **Edge ML & Inference Engine:** `ONNX Runtime` + `skl2onnx`
* **Machine Learning Regressor:** `Scikit-Learn` (Random Forest Regressor)
* **Natural Language Processing (NLP):** `spaCy` (Named Entity Recognition)
* **Data Processing & Analysis:** `Pandas`, `NumPy`
* **Development & Client Platform:** `Ubuntu WSL` (Linux Edge Environment)

---

## ⚙️ Quick Setup

1. **Set Up Virtual Environment & Dependencies:**
   ```bash
   bash setup-environment.sh
   source venv/bin/activate
   ```

2. **Add Dataset:**
   Place `laptops.csv` inside the `datasets/` folder.

---

## 🚀 How to Run (5 Steps)

### Step 1: Clean Data
```bash
python3 preprocess_data-v2.py --input datasets/laptops.csv --output cleaned_laptops.csv
```

### Step 2: Prepare NLP Training Data
```bash
python3 prepare_ner_dataset.py
```

### Step 3: Train & Export ONNX Model
```bash
python3 train_valuation_onnx.py --input cleaned_laptops.csv --output valuation_model.onnx
```

### Step 4: Start Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
> Interactive API Docs available at: `http://localhost:8000/docs`

### Step 5: Run Client Test
In a separate terminal tab:
```bash
python3 client.py --text "Dell XPS 15 laptop with Intel Core i7, 16GB RAM, RTX 3060 GPU, and 1TB SSD"
```

---

## 💡 Output Example

```json
{
  "extracted_specs": {
    "ram_gb": 16.0,
    "storage_gb": 1024.0,
    "cpu_brand": "Intel",
    "gpu_brand": "NVIDIA"
  },
  "estimated_price_usd": 1559.60,
  "price_range_min": 1372.45,
  "price_range_max": 1746.75,
  "performance_tier": "Mid-Range / Business Workhorse Laptop"
}
```
