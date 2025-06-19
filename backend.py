import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Konfiguration ---
SEQ_LEN = 64
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FFN_DIM = 256
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Trainingsdaten (Mini-Trainings-Corpus) ---
text = (
    "Ich heiße Günther. Ich programmiere gerne. "
    "Ich liebe es, neue Dinge zu lernen. Hallo Welt! "
    "Ich habe einen IQ von 150. Programmieren macht Spaß."
)

chars = sorted(list(set(text)))
vocab_size = len(chars)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [char2idx[c] for c in s if c in char2idx]

def decode(l):
    return ''.join(idx2char[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# --- Transformer Komponenten ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(S, S, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, S, D)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# --- Modell ---
class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = PositionalEncoding(EMBED_DIM, SEQ_LEN)
        self.layers = nn.ModuleList([TransformerBlock(EMBED_DIM, NUM_HEADS, FFN_DIM) for _ in range(NUM_LAYERS)])
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, idx):
        x = self.token_emb(idx)
        x = self.pos_emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

model = MiniTransformer().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# --- Training ---
def train_model(epochs=30, batches_per_epoch=100):
    for epoch in range(epochs):
        model.train()
        for _ in range(batches_per_epoch):
            ix = torch.randint(len(data) - SEQ_LEN - 1, (16,))
            x = torch.stack([data[i:i+SEQ_LEN] for i in ix]).to(DEVICE)
            y = torch.stack([data[i+1:i+SEQ_LEN+1] for i in ix]).to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# --- Textgenerierung ---
@torch.no_grad()
def generate(prompt, max_len=200, temperature=1.0):
    model.eval()
    idxs = encode(prompt)
    for _ in range(max_len):
        x = torch.tensor([idxs[-SEQ_LEN:]], dtype=torch.long).to(DEVICE)
        logits = model(x)[0, -1] / temperature
        probs = F.softmax(logits, dim=0)
        next_id = torch.multinomial(probs, 1).item()
        idxs.append(next_id)
    return decode(idxs)

# --- Flask Setup ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/ask", methods=["POST, GET, HEAD"])
def ask():
    data = request.get_json()
    prompt = data.get("message", "").strip()
    if not prompt:
        return jsonify({"reply": "Kein Prompt erhalten."})
    reply = generate(prompt, max_len=300, temperature=0.9)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    print("Training läuft...")
    train_model()
    print("Training abgeschlossen. Starte Server...")
    app.run(port=5000)

