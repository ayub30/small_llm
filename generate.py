import torch
import torch.nn.functional as F
from model import CharModel
from data import encode, decode

device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 128

model = CharModel()
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()

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

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    prompt = f"Q: {question.lower()}\nA:"
    print(f"Bot: {generate(prompt)}")
    print()