from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "dataset" / "processed" / "prepared_resume_dataset.csv"
MODEL_DIR = ROOT / "models"
TEXT_COLUMN = "Processed_Text"
TARGET_COLUMN = "Category"
RANDOM_STATE = 42

def main():
    print("=" * 65)
    print("MEMBER 2 - SVM RESUME CLASSIFICATION")
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
    df = df[(df[TEXT_COLUMN] != "") & (df[TARGET_COLUMN] != "")]
    df = df.drop_duplicates()

    X_train, X_test, y_train_text, y_test_text = train_test_split(
        df[TEXT_COLUMN], df[TARGET_COLUMN],
        test_size=0.20, random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN]
    )

    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        min_df=1, sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_text)
    y_test = encoder.transform(y_test_text)

    print(f"Dataset shape: {df.shape}")
    print(f"TF-IDF training shape: {X_train_tfidf.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nTraining Linear SVM...")
    model = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 65)
    print("SVM RESULTS")
    print("=" * 65)
    print(f"Accuracy: {accuracy:.4f}\n")
    print(classification_report(
        y_test, predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    ))

    joblib.dump(model, MODEL_DIR / "svm_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "svm_tfidf.pkl")
    joblib.dump(encoder, MODEL_DIR / "svm_label_encoder.pkl")

    print("MODEL FILES SAVED")
    print(MODEL_DIR / "svm_model.pkl")
    print(MODEL_DIR / "svm_tfidf.pkl")
    print(MODEL_DIR / "svm_label_encoder.pkl")
    print("\nSUCCESS!")

if __name__ == "__main__":
    main()
