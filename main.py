from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# Allow browser requests from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

class Prompt(BaseModel):
    text: str

# Serve chat UI
@app.get("/")
def serve_ui():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "Chat server running"}

# Chat endpoint
@app.post("/ask")
def ask_ai(p: Prompt):
    payload = {"inputs": p.text}
    try:
        response = requests.post(HF_MODEL_URL, json=payload)
        result = response.json()
        return {"response": result}
    except Exception as e:
        return {"error": str(e)}
