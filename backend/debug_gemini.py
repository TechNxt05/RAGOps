import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)
key = os.getenv("GEMINI_API_KEY")

print(f"Key loaded: {key[:5]}... ({len(key) if key else 0} chars)")

if not key:
    print("No key found.")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={key}"
data = {
    "model": "models/embedding-001",
    "content": {
        "parts": [{"text": "Hello world"}]
    }
}

print("Sending request to Google...")
try:
    resp = requests.post(url, json=data, headers={"Content-Type": "application/json"})
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        print("✅ SUCCESS! Key is valid.")
        print("Embedding:", resp.json().get('embedding', {}).get('values')[:3], "...")
    else:
        print("❌ FAILURE!")
        print(resp.text)
except Exception as e:
    print("❌ EXCEPTION:", e)
