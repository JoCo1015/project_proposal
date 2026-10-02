import requests
import json
import argparse

def test_api(server_url: str, text_prompt: str):
    endpoint = f"{server_url}/predict"
    payload = {"raw_text": text_prompt}

    print(f"\n--- EdgeSpecs Client Sending Request ---")
    print(f"Target Endpoint: {endpoint}")
    print(f"User Input Text: \"{text_prompt}\"\n")

    try:
        response = requests.post(endpoint, json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("--- Inference Response Received (HTTP 200 OK) ---")
            print(json.dumps(data, indent=2))
        else:
            print(f"Error HTTP {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"[Connection Error] Could not connect to FastAPI server at {server_url}.")
        print("Make sure you started the server first using: uvicorn main:app --reload")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test EdgeSpecs Client REST API")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="FastAPI Server URL")
    parser.add_argument("--text", type=str, default="Selling my PC with Intel i7-12700K, 16GB RAM, RTX 3060 GPU, and 1TB SSD", help="Text hardware specs")
    args = parser.parse_args()

    test_api(args.url, args.text)
