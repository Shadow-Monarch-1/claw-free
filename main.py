from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class Prompt(BaseModel):
    text: str

HUGGING_FACE_API = ""  # leave blank for free public inference

@app.post("/ask")
def ask_model(p: Prompt):
    url = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API}"}  # empty for public
    r = requests.post(url, headers=headers, json={"inputs": p.text})
    return r.json()

@app.get("/")
def index():
    return {"status": "running"}
