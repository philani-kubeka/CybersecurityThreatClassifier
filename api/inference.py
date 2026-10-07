import json, re
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
def tokenize(t):
    return re.findall(r"[a-z0-9_]+|[^\sa-z0-9_]", t.lower())
class BiLSTMClassifier(nn.Module): # <- copy EXACTLY from the notebook
    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=128, layers=2,
                dropout=0.4, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
        dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden * 4, 128)
        self.fc2 = nn.Linear(128, n_classes)
    def forward(self, x):
        mask = (x != self.pad_idx).unsqueeze(-1)
        lengths = mask.squeeze(-1).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.emb(x)), lengths,
        batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mx = out.masked_fill(~mask, -1e9).max(dim=1).values
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        h = torch.cat([mx, mean], dim=1)
        return self.fc2(self.drop(torch.relu(self.fc1(self.drop(h)))))

class Predictor:
    def __init__(self, art_dir):
        art = Path(art_dir)
        pre = json.loads((art / "preprocess.json").read_text())
        self.itos, self.max_len, self.classes = pre["itos"], pre["max_len"], pre["classes"]
        self.use_span, self.ctx = pre["use_span"], pre["context_chars"]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        ckpt = torch.load(art / "model.pt", map_location="cpu")
        self.model = BiLSTMClassifier(len(self.itos), len(self.classes), **ckpt["config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
    def make_input(self, t, s=None, e=None): # same logic as make_input() in the notebook
        t = str(t)
        if self.use_span and s is not None and e is not None:
            return f"{t[max(0, s - self.ctx):s]} entstart {t[s:e]} entend {t[e:e + self.ctx]}"
        return t
    @torch.no_grad()
    def _probs(self, batch):
        x = torch.tensor([i + [0] * (self.max_len - len(i)) for i in batch], dtype=torch.long)
        return torch.softmax(self.model(x), 1).numpy()
    def predict(self, text, start=None, end=None, explain=True):
        s = re.sub(r"\s+", " ", self.make_input(text, start, end)).strip()
        toks = tokenize(s)[: self.max_len]
        ids = [self.stoi.get(w, 1) for w in toks]
        base = self._probs([ids])[0]
        k = int(base.argmax())
        out = {"label": self.classes[k], "confidence": float(base[k]),
                "probabilities": {c: float(p) for c, p in zip(self.classes, base)}}
        if explain and ids: # occlusion explanation (see Part 5)
            variants = [ids[:i] + [1] + ids[i + 1:] for i in range(len(ids))]
            drops = base[k] - self._probs(variants)[:, k]
            out["explanation"] = [{"token": t, "score": float(d)} for t, d in zip(toks, drops)]
        return out