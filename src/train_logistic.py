import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# Project location
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# --------------------------------------------------
# Load prepared dataset
# --------------------------------------------------

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "prepared_resume_dataset.csv"
)

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Input and target
# --------------------------------------------------

X_text = df["Processed_Text"].fillna("")

y = df["Category"]


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X_text)


print("TF-IDF shape:", X.shape)


# --------------------------------------------------
# Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000
)


print()
print("Training Logistic Regression...")

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

predictions = model.predict(
    X_test
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

print()
print("===================================")
print("LOGISTIC REGRESSION RESULTS")
print("===================================")

print(
    "Accuracy:",
    round(accuracy, 4)
)

print()
print(
    classification_report(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# Save model
# --------------------------------------------------

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "logistic_regression.pkl"
    )
)


joblib.dump(
    vectorizer,
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )
)


print()
print("Model saved successfully.")