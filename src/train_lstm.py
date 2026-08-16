import os
import re
import json
import torch
import torch.nn as nn

import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "prepared_resume_dataset.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 300

EMBEDDING_DIM = 128
HIDDEN_DIM = 128

BATCH_SIZE = 32
EPOCHS = 3

LEARNING_RATE = 0.001


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("LSTM RESUME CLASSIFICATION")
print("=" * 60)

print()
print("Using device:", device)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# SELECT TEXT
# ============================================================

texts = df["Processed_Text"].fillna("").astype(str).tolist()

labels = df["Category"].astype(str).tolist()


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(labels)

num_classes = len(label_encoder.classes_)

print()
print("Number of classes:", num_classes)

print("Classes:")
print(list(label_encoder.classes_))


# ============================================================
# BUILD VOCABULARY
# ============================================================

print()
print("Building vocabulary...")


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


counter = Counter()

for text in texts:
    counter.update(tokenize(text))


most_common_words = counter.most_common(MAX_VOCAB_SIZE - 2)


word_to_index = {
    "<PAD>": 0,
    "<UNK>": 1
}


for index, (word, count) in enumerate(most_common_words, start=2):
    word_to_index[word] = index


print("Vocabulary size:", len(word_to_index))


# ============================================================
# CONVERT TEXT TO SEQUENCES
# ============================================================

def text_to_sequence(text):

    words = tokenize(text)

    sequence = []

    for word in words[:MAX_SEQUENCE_LENGTH]:

        if word in word_to_index:
            sequence.append(word_to_index[word])
        else:
            sequence.append(word_to_index["<UNK>"])

    # Padding

    while len(sequence) < MAX_SEQUENCE_LENGTH:
        sequence.append(word_to_index["<PAD>"])

    return sequence


print()
print("Converting resumes into sequences...")


X = torch.tensor(
    [text_to_sequence(text) for text in texts],
    dtype=torch.long
)

y = torch.tensor(
    encoded_labels,
    dtype=torch.long
)


print("Input shape:", X.shape)
print("Label shape:", y.shape)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            hidden_dim,
            num_classes
        )


    def forward(self, x):

        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(embedded)

        last_hidden = hidden[-1]

        last_hidden = self.dropout(last_hidden)

        output = self.fc(last_hidden)

        return output


# ============================================================
# CREATE MODEL
# ============================================================

model = LSTMClassifier(
    vocab_size=len(word_to_index),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=num_classes
).to(device)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 60)
print("TRAINING LSTM")
print("=" * 60)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    permutation = torch.randperm(
        X_train.size(0)
    )

    for i in range(
        0,
        X_train.size(0),
        BATCH_SIZE
    ):

        indices = permutation[
            i:i + BATCH_SIZE
        ]

        batch_x = X_train[indices].to(device)

        batch_y = y_train[indices].to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    average_loss = total_loss / (
        X_train.size(0) / BATCH_SIZE
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 60)
print("LSTM EVALUATION")
print("=" * 60)


model.eval()

all_predictions = []

all_actual = []


with torch.no_grad():

    for i in range(
        0,
        X_test.size(0),
        BATCH_SIZE
    ):

        batch_x = X_test[
            i:i + BATCH_SIZE
        ].to(device)

        outputs = model(batch_x)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_actual.extend(
            y_test[
                i:i + BATCH_SIZE
            ].numpy()
        )


accuracy = accuracy_score(
    all_actual,
    all_predictions
)


print()
print("LSTM Accuracy:", round(accuracy, 4))

print()
print("Classification Report:")

print(
    classification_report(
        all_actual,
        all_predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "lstm_model.pth"
)

vocab_path = os.path.join(
    MODEL_DIR,
    "lstm_vocab.json"
)

labels_path = os.path.join(
    MODEL_DIR,
    "lstm_labels.json"
)


torch.save(
    model.state_dict(),
    model_path
)


with open(
    vocab_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        word_to_index,
        file
    )


with open(
    labels_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        list(label_encoder.classes_),
        file
    )


print()
print("=" * 60)
print("LSTM MODEL SAVED")
print("=" * 60)

print(model_path)
print(vocab_path)
print(labels_path)

print()
print("SUCCESS!")