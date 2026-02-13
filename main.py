from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

# Model endpoint (Hugging Face public free model)
HF_MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

class Prompt(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "Chat server running"}

@app.post("/ask")
def ask_ai(p: Prompt):
    payload = {"inputs": p.text}
    try:
        # Call Hugging Face public inference
        response = requests.post(HF_MODEL_URL, json=payload)
        result = response.json()
        return {"response": result}
    except Exception as e:
        return {"error": str(e)}
