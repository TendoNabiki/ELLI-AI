import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from pathlib import Path
from tokenizers import Tokenizer


if torch.cuda.is_available():
    device = "cuda" 
    print("GPU is avaliable")    
else:
    device = "cpu"
    print("CPU")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

block_size = 128
batch_size = 64

d_model = 256
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

# SMOKE_TEST mode -- runs a handful of steps on your real model size, real
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

TOKENIZER_PATH = "New_vocab.json"
if not Path(TOKENIZER_PATH).exists():
    raise FileNotFoundError(
        f"Missing {TOKENIZER_PATH} -- make sure you're running this script from "
        f"the directory containing your tokenizer file (or update TOKENIZER_PATH)."
    )
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()

def encode(s: str):
    return tokenizer.encode(s).ids

def decode(ids):
    return tokenizer.decode(ids)

# CHANGED: no pre-tokenized .bin files at all anymore. Training reads directly
# from your Training_Data/*.json text files (same format as your tokenizer
# training code: one document per line). To do this without ever loading the
# whole multi-billion-token corpus into RAM, this builds a lightweight INDEX
# of where every line lives on disk (which file, what byte offset) -- not the
# text itself. Each training example is then assembled by seeking to a few
# random offsets, reading just those lines, and tokenizing them fresh, on the
# spot. Nothing about the corpus is ever pre-tokenized or persisted to disk.
JSON_DIR = "Training_Data"
VAL_FRACTION = 0.05  # fraction of documents (by count) held out for val

json_paths = sorted(Path(JSON_DIR).glob("*.json"))
if not json_paths:
    raise FileNotFoundError(f"No .json files found in {JSON_DIR}/")

# One-time scan to find every non-empty line's (file, byte offset). This reads
# through the files but never tokenizes or holds full text in memory -- only
# small integers get kept, so this stays cheap even across ~1000 files.
_file_idx_list = []
_offset_list = []
for fi, path in enumerate(json_paths):
    with open(path, "rb") as f:
        while True:
            offset = f.tell()
            raw_line = f.readline()
            if not raw_line:
                break
            if raw_line.decode("utf-8", errors="replace").strip():
                _file_idx_list.append(fi)
                _offset_list.append(offset)

file_indices = np.array(_file_idx_list, dtype=np.uint32)
byte_offsets = np.array(_offset_list, dtype=np.int64)
print(f"indexed {len(json_paths)} .json files, {len(file_indices):,} documents total")

# Document-level train/val split -- same idea as before (whole documents only
# ever land on one side), just expressed as an index split instead of a
# token-stream split, since there's no token stream anymore.
n_docs = len(file_indices)
n_val = max(1, int(VAL_FRACTION * n_docs)) if VAL_FRACTION > 0 else 0
if n_val > 0:
    train_file_idx, train_offsets = file_indices[:-n_val], byte_offsets[:-n_val]
    val_file_idx, val_offsets = file_indices[-n_val:], byte_offsets[-n_val:]
else:
    train_file_idx, train_offsets = file_indices, byte_offsets
    val_file_idx, val_offsets = np.array([], dtype=np.uint32), np.array([], dtype=np.int64)

# Optional, flexible control over how much data actually gets used -- now
# expressed in documents rather than tokens, since there's no pre-tokenized
# array to slice. Leave both as None to use every indexed document.
DATA_FRACTION = None    # e.g. 0.1 = use only the first 10% of documents in each split
MAX_TRAIN_DOCS = None   # e.g. 50_000 = cap the train split at 50k documents
MAX_VAL_DOCS = None     # same idea, for val

if DATA_FRACTION is not None:
    train_file_idx = train_file_idx[:int(len(train_file_idx) * DATA_FRACTION)]
    train_offsets = train_offsets[:int(len(train_offsets) * DATA_FRACTION)]
    val_file_idx = val_file_idx[:int(len(val_file_idx) * DATA_FRACTION)]
    val_offsets = val_offsets[:int(len(val_offsets) * DATA_FRACTION)]
if MAX_TRAIN_DOCS is not None:
    train_file_idx = train_file_idx[:MAX_TRAIN_DOCS]
    train_offsets = train_offsets[:MAX_TRAIN_DOCS]
if MAX_VAL_DOCS is not None:
    val_file_idx = val_file_idx[:MAX_VAL_DOCS]
    val_offsets = val_offsets[:MAX_VAL_DOCS]

assert len(train_file_idx) > 0, f"no training documents indexed -- check {JSON_DIR}/*.json has non-empty lines"
assert len(val_file_idx) > 0, f"no validation documents indexed -- raise VAL_FRACTION or add more data to {JSON_DIR}/*.json"

bos_id = tokenizer.token_to_id("[BOS]")
eos_id = tokenizer.token_to_id("[EOS]")

def _read_line(file_idx, offset):
    with open(json_paths[file_idx], "rb") as f:
        f.seek(offset)
        raw_line = f.readline()
    return raw_line.decode("utf-8", errors="replace").strip()

def _sample_example(file_idx_arr, offset_arr, target_len):
    # Keep pulling random documents (each wrapped in [BOS]...[EOS], same as
    # your data-prep code) and concatenating them until there are enough
    # tokens to fill one training example, then truncate to exactly
    # target_len. A single example can span multiple documents, same as
    # standard causal LM training on concatenated documents.
    ids = []
    while len(ids) < target_len:
        i = np.random.randint(0, len(file_idx_arr))
        line = _read_line(int(file_idx_arr[i]), int(offset_arr[i]))
        if not line:
            continue
        ids.extend([bos_id] + tokenizer.encode(line).ids + [eos_id])
    return ids[:target_len]

torch.manual_seed(0)

def get_batch(split):
    file_idx_arr, offset_arr = (train_file_idx, train_offsets) if split == "train" else (val_file_idx, val_offsets)
    xs, ys = [], []
    for _ in range(batch_size):
        ids = _sample_example(file_idx_arr, offset_arr, block_size + 1)
        xs.append(ids[:-1])
        ys.append(ids[1:])
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
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

# ADDED: optional warm-start from an existing checkpoint (e.g. one produced by
# grow_model.py, or just resuming these exact dimensions) instead of fresh
# random weights. Leave as None for the previous/default behavior.
# Note: only the model weights are restored here, not the optimizer state --
# for a grown model the old optimizer's per-parameter moment estimates don't
# apply to the newly appended layers anyway, so it's simplest to always start
# the optimizer fresh below regardless of whether this is used.
INIT_FROM_CHECKPOINT = None  # e.g. "checkpoints/grown_model.pt"
if INIT_FROM_CHECKPOINT is not None:
    _init_ckpt = torch.load(INIT_FROM_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(_init_ckpt["model_state_dict"])
    print(f"initialized model weights from {INIT_FROM_CHECKPOINT}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# torch.compile() itself doesn't fail even on an unsupported setup -- it compiles
# lazily on first real call, so this runs one dummy forward pass right now to force
# compilation immediately and fall back to eager mode cleanly if it doesn't work.
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

# bf16 for RTX 5080 (Blackwell -- full native bf16 tensor core support). bf16 has
# the same exponent range as fp32, so unlike fp16 it doesn't need GradScaler /
# loss scaling at all.
amp_dtype = torch.bfloat16

# ADDED: torch.compile wraps the model, so model.state_dict() would save every key
# prefixed with "_orig_mod." -- harmless while training continues in this same
# process, but it silently breaks load_state_dict() later into a fresh, uncompiled
# ELLI (e.g. for inference, or resuming after a code fix). This always saves plain,
# portable keys regardless of whether torch.compile is active.
def raw_state_dict(m):
    return m._orig_mod.state_dict() if hasattr(m, "_orig_mod") else m.state_dict()

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
# CHANGED: back to tracking best val loss (a real generalization signal) rather
# than train loss (which keeps trending down even if the model starts
# memorizing, so it's a weaker signal on its own).
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
                "model_state_dict": raw_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, checkpoint_dir / "best_model.pt")

    xb, yb = get_batch("train")

    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=True):
        _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

torch.save({
    "step": max_steps,
    "model_state_dict": raw_state_dict(model),
    "optimizer_state_dict": optimizer.state_dict(),
}, checkpoint_dir / "final_model.pt")

print(f"training done in {time.time() - start_time:.0f}s, best val loss {best_val_loss:.4f}")