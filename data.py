import torch
import os
import boto3

S3_BUCKET = "inv-llm"
S3_DATA_PREFIX = "data/"

def download_data_from_s3():
    s3 = boto3.client("s3")
    objects = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_DATA_PREFIX)
    all_text = ""
    for obj in objects.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".txt"):
            response = s3.get_object(Bucket=S3_BUCKET, Key=key)
            all_text += response["Body"].read().decode("utf-8") + "\n"
    print(f"Downloaded {len(all_text)} characters from S3")
    return all_text

text = download_data_from_s3()

if not text.strip():
    print("S3 returned no data, falling back to local file")
    with open(os.path.join("data", "input.txt"), "r", encoding="utf-8") as f:
        text = f.read()

text = text * 50

chars = sorted(set(text))
vocab_size = len(chars)

ctoi = {ch: i for i, ch in enumerate(chars)}
itoc = {i: ch for i, ch in enumerate(chars)}

def encode(chars):
    return [ctoi[c] for c in chars]

def decode(indices):
    return "".join(itoc[i] for i in indices)

test = "Q: What is VAT?"
encoded = encode(test)
decoded = decode(encoded)

print(f"original:  {test}")
print(f"encoded:   {encoded}")
print(f"decoded:   {decoded}")
print(f"roundtrip: {test == decoded}")

data = torch.tensor(encode(text), dtype=torch.long)
print("data shape: ", data.shape)
print("data dtype: ", data.dtype)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]