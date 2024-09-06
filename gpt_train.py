import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt_model import GPTLanguageModel
from utils import get_encoder_decoder
from tqdm import tqdm

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
vocab_size = 1118
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.2
# ------------


torch.manual_seed(1337)


with open("data/input.txt", "r", encoding="utf-8") as f:
    finetune_text = f.read()

# Load the tokenizer (the encoding/decoding functions)
encode, decode = get_encoder_decoder(path="data/text_data.txt")


# Train and test splits
data = torch.tensor(encode(finetune_text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = GPTLanguageModel()
model.load_state_dict(
    torch.load(
        "models/gpt_model.pth",
        map_location=device,
        weights_only=True,
    )
)
model.to(device)
print(
    f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f} million parameters"
)
print(f"{device=}")
print(f"{vocab_size=}")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in tqdm(range(max_iters)):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(
    model.state_dict(),
    "models/gpt_model_shakespeare_fine_tuned.pth",
)

# generate from the model

context = torch.tensor([encode(f"MARCIUS:\n")], dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=256)[0].tolist()))
