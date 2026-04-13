import torch
import os

with open(os.path.join("data", "train.txt"), "r", encoding="utf-8") as f:
    text = f.read()


chars = sorted(set(text))
vocab_size = len(chars)

ctoi = {ch: i for i, ch in enumerate(chars)}
itoc = {i: ch for i, ch in enumerate(chars)}

def encode(chars):
    return [ctoi[c] for c in chars]

def decode(indices):
    return "".join(itoc[i] for i in indices)

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


block_size = 64
batch_size = 32

def get_batch(split):
    pool = train_data if split == "train" else val_data
    start_index = torch.randint(len(pool) - block_size, (batch_size,))
    x = torch.stack([pool[i:i+block_size] for i in start_index])
    y = torch.stack([pool[i+1:i+block_size+1] for i in start_index])
    return x, y