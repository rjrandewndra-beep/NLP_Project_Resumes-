"""CPU-friendly fine-tuning for a real, compact BERT text classifier."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "dataset" / "processed" / "prepared_resume_dataset.csv"
OUTPUT_DIR = ROOT / "models" / "bert_resume_classifier"
MODEL_NAME = "prajjwal1/bert-tiny"


class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    return actual, predicted


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    df = pd.read_csv(args.data).dropna(subset=["Processed_Text", "Category"])
    if args.max_samples and args.max_samples < len(df):
        # Keep every class represented while limiting CPU work.
        df, _ = train_test_split(
            df,
            train_size=args.max_samples,
            random_state=args.seed,
            stratify=df["Category"],
        )

    encoder = LabelEncoder()
    labels = encoder.fit_transform(df["Category"].astype(str))
    train_text, test_text, train_y, test_y = train_test_split(
        df["Processed_Text"].astype(str),
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    id2label = {i: label for i, label in enumerate(encoder.classes_)}
    label2id = {label: i for i, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(encoder.classes_),
        id2label=id2label,
        label2id=label2id,
    )
    device = torch.device("cpu")
    model.to(device)

    train_loader = DataLoader(
        ResumeDataset(train_text, train_y, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        ResumeDataset(test_text, test_y, tokenizer, args.max_length),
        batch_size=args.batch_size,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_loss += output.loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {total_loss / max(1, step):.4f}")

    actual, predicted = evaluate(model, test_loader, device)
    report = classification_report(
        actual,
        predicted,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": args.model_name,
        "configuration": "CPU demonstration",
        "samples_used": int(len(df)),
        "train_samples": int(len(train_y)),
        "test_samples": int(len(test_y)),
        "categories": int(len(encoder.classes_)),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_weighted": float(report["weighted avg"]["precision"]),
        "recall_weighted": float(report["weighted avg"]["recall"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output / "labels.json").write_text(
        json.dumps({str(i): label for i, label in id2label.items()}, indent=2),
        encoding="utf-8",
    )
    (args.output / "bert_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(report).transpose().to_csv(args.output / "classification_report.csv")
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
