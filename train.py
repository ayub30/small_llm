import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import boto3
import os
from model import CharModel
from data import encode, decode, train_data, val_data, vocab_size

device = "cuda" if torch.cuda.is_available() else "cpu"

S3_BUCKET = "inv-llm"
S3_MODEL_KEY = "models/model_latest.pt"
S3_DATA_PREFIX = "data/"

block_size = 128
n_embd     = 128
n_head     = 4
n_layer    = 4
dropout    = 0.1
batch_size = 32
learning_rate = 1e-3
max_iters = 5000
eval_interval = 500
eval_iters = 200

def upload_model_to_s3():
    s3 = boto3.client("s3")
    s3.upload_file("model.pt", S3_BUCKET, S3_MODEL_KEY)
    print(f"Model uploaded to s3://{S3_BUCKET}/{S3_MODEL_KEY}")

model = CharModel().to(device)

def get_batch(split):
    pool = train_data if split == "train" else val_data
    start_index = torch.randint(len(pool) - block_size, (batch_size,))
    x = torch.stack([pool[i:i+block_size] for i in start_index])
    y = torch.stack([pool[i+1:i+block_size+1] for i in start_index])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits = model(x)
            batch, seq_len, vocab_size = logits.shape
            logits = logits.view(batch * seq_len, vocab_size)
            y = y.view(batch * seq_len)
            loss = F.cross_entropy(logits, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

mlflow.set_experiment("invoice-llm")

with mlflow.start_run():
    mlflow.log_params({
        "vocab_size": vocab_size,
        "block_size": block_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "max_iters": max_iters,
        "batch_size": batch_size
    })

    for i in range(max_iters):

        if i % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            mlflow.log_metrics({
                "train_loss": losses["train"].item(),
                "val_loss": losses["val"].item()
            }, step=i)

        x, y = get_batch("train")
        logits = model(x)
        batch, seq_len, vocab_size = logits.shape
        logits = logits.view(batch * seq_len, vocab_size)
        y = y.view(batch * seq_len)
        loss = F.cross_entropy(logits, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    torch.save(model.state_dict(), "model.pt")
    print("model saved to model.pt")
    upload_model_to_s3()
    mlflow.log_artifact("model.pt")
    mlflow.log_metric("final_val_loss", losses["val"].item())