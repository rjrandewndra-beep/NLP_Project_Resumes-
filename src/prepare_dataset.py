import pandas as pd
import os

# --------------------------------------------------
# 1. Find the project folder
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# --------------------------------------------------
# 2. Dataset path
# --------------------------------------------------

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "resume_dataset.csv"
)

# --------------------------------------------------
# 3. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATASET_PATH)

print("Original dataset shape:", df.shape)

# --------------------------------------------------
# 4. Remove duplicate resumes
# --------------------------------------------------

df = df.drop_duplicates(subset="ID")

# --------------------------------------------------
# 5. Fill missing text
# --------------------------------------------------

df["Cleaned_Text"] = df["Cleaned_Text"].fillna(
    df["Resume_str"]
)

df["Processed_Text"] = df["Processed_Text"].fillna(
    df["Cleaned_Text"]
)

# --------------------------------------------------
# 6. Remove empty resumes
# --------------------------------------------------

df = df[
    df["Resume_str"].notna()
]

# --------------------------------------------------
# 7. Save prepared dataset
# --------------------------------------------------

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "prepared_resume_dataset.csv"
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Prepared dataset shape:", df.shape)

print()
print("Categories:")
print(df["Category"].value_counts())

print()
print("Prepared dataset saved to:")
print(OUTPUT_PATH)