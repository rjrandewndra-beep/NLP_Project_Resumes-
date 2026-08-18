"""Train and save the required TF-IDF + Random Forest resume classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "dataset" / "processed" / "prepared_resume_dataset.csv"
MODEL_DIR = ROOT / "models"


def train(data_path: Path, test_size: float = 0.2, random_state: int = 42) -> dict:
    df = pd.read_csv(data_path)
    required = {"Processed_Text", "Category"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df = df.dropna(subset=["Processed_Text", "Category"]).copy()
    texts = df["Processed_Text"].astype(str)
    labels = df["Category"].astype(str)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        stop_words="english",
    )
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model = RandomForestClassifier(
        # 150 trees keeps the deployable artifact comfortably below GitHub's
        # single-file limit while retaining strong demonstration accuracy.
        n_estimators=150,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train_tfidf, y_train)
    predictions = model.predict(x_test_tfidf)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    metrics = {
        "model": "TF-IDF + Random Forest",
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "categories": int(len(encoder.classes_)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "random_state": random_state,
    }
    report = classification_report(
        y_test,
        predictions,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "random_forest_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")
    (MODEL_DIR / "random_forest_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(report).transpose().to_csv(
        MODEL_DIR / "random_forest_classification_report.csv"
    )

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved artifacts in: {MODEL_DIR}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.test_size, args.random_state)
