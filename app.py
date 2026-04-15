from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import boto3
import os
from model import CharModel
from data import encode, decode

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 128

S3_BUCKET = "inv-llm"
S3_MODEL_KEY = "models/model_latest.pt"

def download_model_from_s3():
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, "model.pt")
    print(f"Model downloaded from s3://{S3_BUCKET}/{S3_MODEL_KEY}")

download_model_from_s3()

model = CharModel()
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()

class Question(BaseModel):
    text: str

@torch.no_grad()
def generate(prompt, max_new_tokens=100):
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        context_cropped = context[:, -block_size:]
        logits = model(context_cropped)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=1)
    response = decode(context[0].tolist())
    answer = response[len(prompt):]
    return answer.split("\n\n")[0]

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/ask")
def ask(question: Question):
    prompt = f"Q: {question.text.lower()}\nA:"
    answer = generate(prompt)
    return {"question": question.text, "answer": answer}