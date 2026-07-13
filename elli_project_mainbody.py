import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import math
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
from tokenizers import Tokenizer  # ADDED: your real BPE tokenizer, replacing the byte-level stand-in

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

block_size = 128
batch_size = 64

d_model = 1024
n_heads = 16
n_layers = 20
dropout = 0.1

max_steps = 1000
lr = 1e-4
weight_decay = 0.1
grad_clip = 1.0
eval_every = 200
eval_iters = 30

warmup_steps = 100
min_lr_ratio = 0.1

def get_lr(step):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    min_lr = lr * min_lr_ratio
    return min_lr + coeff * (lr - min_lr)

# ADDED: SMOKE_TEST mode -- runs a handful of steps on your real model size, real
# block_size, and real batch_size (so it still tells you whether the model fits in
# your RTX 5080's VRAM), just with far fewer steps so you get a pass/fail answer in
# seconds instead of committing to the full run blind. Flip to False for the real run.
SMOKE_TEST = True
if SMOKE_TEST:
    max_steps = 20
    eval_every = 5
    eval_iters = 2
    warmup_steps = 5
    print("=== SMOKE_TEST=True: quick sanity pass on the real model/data, not a real training run ===")

# --- CHANGED: real tokenizer instead of byte-level.
tokenizer = Tokenizer.from_file("my_custom_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()  # was hardcoded 256, now pulled from your actual tokenizer

def encode(s: str):
    return tokenizer.encode(s).ids

def decode(ids):
    return tokenizer.decode(ids)

# --- CHANGED: reads train.bin / val.bin directly, produced by prepare_bin_data.py
# (see that script -- it encodes Training_Data/*.json with my_custom_tokenizer.json
# and saves uint16 token ids, which fits since your vocab_size=10000 < 65536).
# If you already have your own .bin-writing script with a different layout
# (different dtype, padding, etc.), tell me and I'll match it exactly instead.
train_data = torch.from_numpy(np.fromfile("Training_Data/train.bin", dtype=np.uint16).astype(np.int64))
val_data = torch.from_numpy(np.fromfile("Training_Data/val.bin", dtype=np.uint16).astype(np.int64))

# ADDED: optional, flexible control over how much data actually gets used. Leave
# both as None to use the full .bin files (previous/default behavior). Set
# DATA_FRACTION for a quick percentage-based subset, or MAX_*_TOKENS for a hard
# token-count cap -- either works standalone, or combine them (fraction applied
# first, then the cap).
DATA_FRACTION = None     # e.g. 0.1 = use only the first 10% of tokens in each split
MAX_TRAIN_TOKENS = None  # e.g. 2_000_000 = cap train.bin at 2M tokens
MAX_VAL_TOKENS = None    # same idea, for val.bin

if DATA_FRACTION is not None:
    train_data = train_data[:int(len(train_data) * DATA_FRACTION)]
    val_data = val_data[:int(len(val_data) * DATA_FRACTION)]
if MAX_TRAIN_TOKENS is not None:
    train_data = train_data[:MAX_TRAIN_TOKENS]
if MAX_VAL_TOKENS is not None:
    val_data = val_data[:MAX_VAL_TOKENS]

assert len(train_data) > block_size + 1, "train.bin has fewer tokens than block_size -- check prepare_bin_data.py ran on your full corpus"
assert len(val_data) > block_size + 1, "val.bin has fewer tokens than block_size -- corpus too small for this block_size"

torch.manual_seed(0)

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out

# Model
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads
        Hd = self.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, H, Hd).transpose(1, 2)  # (B,H,T,Hd)
        k = k.view(B, T, H, Hd).transpose(1, 2)
        v = v.view(B, T, H, Hd).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(Hd)  # (B,H,T,T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        att = F.softmax(scores, dim=-1)
        att = self.drop(att)

        out = att @ v  # (B,H,T,Hd)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.drop(self.out(out))
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, block_size)
        self.ff = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class ELLI(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, block_size)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)  # (B,T,D)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T,D)
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx

# Train

model = ELLI(vocab_size, d_model, n_heads, n_layers, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# CHANGED: torch.compile() itself doesn't fail even on an unsupported setup -- it
# compiles lazily on first real call, so wrapping only the compile() line in
# try/except wouldn't have caught a broken setup until deep into your first real
# training step. This runs one dummy forward pass right now to force compilation
# immediately and fall back to eager mode cleanly if it doesn't work.
try:
    compiled_model = torch.compile(model)
    with torch.no_grad():
        _dummy_x, _dummy_y = get_batch("train")
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
            compiled_model(_dummy_x, _dummy_y)
    model = compiled_model
    print("torch.compile: enabled")
except Exception as e:
    print(f"torch.compile: failed, falling back to eager mode ({e})")

# CHANGED: bf16 instead of fp16, per your RTX 5080 (Blackwell -- full native bf16
# tensor core support). bf16 has the same exponent range as fp32, so unlike fp16 it
# doesn't need GradScaler / loss scaling at all -- that machinery is removed below.
amp_dtype = torch.bfloat16

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
best_val_loss = float("inf")

print("device:", device)
print("vocab_size:", vocab_size)
print("params:", sum(p.numel() for p in model.parameters()))

start_time = time.time()

for step in range(max_steps + 1):
    lr_now = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr_now

    if step % eval_every == 0:
        losses = estimate_loss(model)
        elapsed = time.time() - start_time
        print(f"step {step} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr_now:.2e} | {elapsed:.0f}s")

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, checkpoint_dir / "best_model.pt")

    xb, yb = get_batch("train")

    # CHANGED: bf16 autocast, no GradScaler needed (see note above).
    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=True):
        _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

torch.save({
    "step": max_steps,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, checkpoint_dir / "final_model.pt")

print(f"training done in {time.time() - start_time:.0f}s, best val loss {best_val_loss:.4f}")
