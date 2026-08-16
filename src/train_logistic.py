import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "prepared_resume_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOGISTIC REGRESSION TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# REMOVE EMPTY TEXT
# ============================================================

df["Processed_Text"] = df["Processed_Text"].fillna("")

df = df[
    df["Processed_Text"].str.strip() != ""
]


# ============================================================
# TEXT DATA
# ============================================================

texts = df["Processed_Text"]

labels = df["Category"]


# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X = vectorizer.fit_transform(texts)

print("TF-IDF shape:", X.shape)


# ============================================================
# ENCODE LABELS
# ============================================================

encoder = LabelEncoder()

y = encoder.fit_transform(labels)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=2000
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# EVALUATION
# ============================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# SAVE MODELS
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "logistic_model.pkl"
)

tfidf_path = os.path.join(
    MODEL_DIR,
    "tfidf.pkl"
)

encoder_path = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)


joblib.dump(
    model,
    model_path
)

joblib.dump(
    vectorizer,
    tfidf_path
)

joblib.dump(
    encoder,
    encoder_path
)


print("\n" + "=" * 60)
print("MODEL FILES SAVED")
print("=" * 60)

print(model_path)
print(tfidf_path)
print(encoder_path)

print("\nSUCCESS!")