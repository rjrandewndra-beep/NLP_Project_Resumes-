from pathlib import Path
import json
import re
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "gru_model.pth"
VOCAB_FILE = MODEL_DIR / "gru_vocab.json"
LABELS_FILE = MODEL_DIR / "gru_labels.json"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

class ResumeGRU(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=96, num_layers=1, dropout=0.30):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, hidden = self.gru(self.embedding(x))
        return self.classifier(self.dropout(hidden[-1]))

def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())

def _load():
    for path in (MODEL_FILE, VOCAB_FILE, LABELS_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required GRU file not found: {path}")

    checkpoint = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
    with open(VOCAB_FILE, encoding="utf-8") as f:
        vocab = json.load(f)
    with open(LABELS_FILE, encoding="utf-8") as f:
        labels = json.load(f)

    model = ResumeGRU(
        checkpoint["vocab_size"], checkpoint["num_classes"],
        checkpoint["embedding_dim"], checkpoint["hidden_dim"],
        checkpoint["num_layers"], checkpoint["dropout"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocab, labels, checkpoint["max_length"]

def predict_resume(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Resume text must be a non-empty string.")

    model, vocab, labels, max_length = _load()
    ids = [vocab.get(t, vocab.get(UNK_TOKEN, 1)) for t in tokenize(text)][:max_length]
    ids += [vocab.get(PAD_TOKEN, 0)] * (max_length - len(ids))
    X = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        probabilities = torch.softmax(model(X), dim=1)[0]
        predicted_id = int(torch.argmax(probabilities))

    return {
        "category": str(labels[predicted_id]),
        "confidence_score": float(probabilities[predicted_id])
    }

def predict_resumes(texts):
    return [predict_resume(text) for text in texts]

if __name__ == "__main__":
    sample = "Python developer with SQL, machine learning, pandas and data analysis experience."
    print(predict_resume(sample))
