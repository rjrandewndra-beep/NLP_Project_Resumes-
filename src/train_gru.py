from pathlib import Path
import json
import random
import re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "dataset" / "processed" / "prepared_resume_dataset.csv"
MODEL_DIR = ROOT / "models"
TEXT_COLUMN = "Processed_Text"
TARGET_COLUMN = "Category"
RANDOM_STATE = 42

MAX_VOCAB_SIZE = 12000
MAX_LENGTH = 300
EMBEDDING_DIM = 128
HIDDEN_DIM = 96
NUM_LAYERS = 1
DROPOUT = 0.30
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.001

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())

def build_vocab(texts):
    counts = {}
    for text in texts:
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in ordered[:MAX_VOCAB_SIZE - 2]:
        vocab[token] = len(vocab)
    return vocab

def encode_text(text, vocab):
    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokenize(text)][:MAX_LENGTH]
    ids += [vocab[PAD_TOKEN]] * (MAX_LENGTH - len(ids))
    return ids

class ResumeGRU(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.gru = nn.GRU(
            EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        return self.classifier(self.dropout(hidden[-1]))

def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    preds, labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(y)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    loss = total_loss / max(1, len(labels))
    return loss, accuracy_score(labels, preds), np.array(labels), np.array(preds)

def main():
    set_seed()
    print("=" * 65)
    print("MEMBER 2 - GRU RESUME CLASSIFICATION")
    print("=" * 65)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found:\n{DATA_FILE}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    missing = {TEXT_COLUMN, TARGET_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[[TEXT_COLUMN, TARGET_COLUMN]].copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna("").astype(str).str.strip()
    df = df[(df[TEXT_COLUMN] != "") & (df[TARGET_COLUMN] != "")].drop_duplicates()

    train_text, test_text, train_labels, test_labels = train_test_split(
        df[TEXT_COLUMN].tolist(), df[TARGET_COLUMN].tolist(),
        test_size=0.20, random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN]
    )

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_labels)
    y_test = encoder.transform(test_labels)

    vocab = build_vocab(train_text)
    X_train = np.array([encode_text(t, vocab) for t in train_text], dtype=np.int64)
    X_test = np.array([encode_text(t, vocab) for t in test_text], dtype=np.int64)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset shape: {df.shape}")
    print(f"Training samples: {len(train_text)}")
    print(f"Testing samples: {len(test_text)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of classes: {len(encoder.classes_)}")
    print(f"Using device: {device}")

    model = ResumeGRU(len(vocab), len(encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining GRU...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * len(y)
            seen += len(y)

        train_loss = running_loss / max(1, seen)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    _, test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    print("\n" + "=" * 65)
    print("GRU RESULTS")
    print("=" * 65)
    print(f"Accuracy: {test_acc:.4f}\n")
    print(classification_report(
        y_true, y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    ))

    torch.save({
        "state_dict": model.state_dict(),
        "vocab_size": len(vocab),
        "num_classes": len(encoder.classes_),
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "max_length": MAX_LENGTH,
    }, MODEL_DIR / "gru_model.pth")

    with open(MODEL_DIR / "gru_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(MODEL_DIR / "gru_labels.json", "w", encoding="utf-8") as f:
        json.dump(encoder.classes_.tolist(), f, ensure_ascii=False)

    print("\nMODEL FILES SAVED")
    print(MODEL_DIR / "gru_model.pth")
    print(MODEL_DIR / "gru_vocab.json")
    print(MODEL_DIR / "gru_labels.json")
    print("\nSUCCESS!")

if __name__ == "__main__":
    main()
