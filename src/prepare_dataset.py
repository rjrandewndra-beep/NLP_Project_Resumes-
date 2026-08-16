import os
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# Possible locations of the datasets
POSSIBLE_RESUME_FILES = [
    os.path.join(BASE_DIR, "dataset", "resumes", "Resume.csv"),
    os.path.join(BASE_DIR, "dataset", "Resume.csv"),
]

POSSIBLE_PROCESSED_FILES = [
    os.path.join(BASE_DIR, "dataset", "resumes", "resume_dataset.csv"),
    os.path.join(BASE_DIR, "dataset", "resume_dataset.csv"),
]


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "processed"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "prepared_resume_dataset.csv"
)


# ============================================================
# FIND FILE
# ============================================================

def find_file(possible_files):

    for file_path in possible_files:

        if os.path.exists(file_path):
            return file_path

    return None


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("RESUME DATASET PREPARATION")
print("=" * 60)


# ------------------------------------------------------------
# Find Resume.csv
# ------------------------------------------------------------

resume_file = find_file(POSSIBLE_RESUME_FILES)

if resume_file is None:

    print("\nERROR: Resume.csv was not found.")

    print("\nI checked these locations:")

    for path in POSSIBLE_RESUME_FILES:
        print(path)

    raise SystemExit


# ------------------------------------------------------------
# Find resume_dataset.csv
# ------------------------------------------------------------

processed_file = find_file(POSSIBLE_PROCESSED_FILES)

if processed_file is None:

    print("\nERROR: resume_dataset.csv was not found.")

    print("\nI checked these locations:")

    for path in POSSIBLE_PROCESSED_FILES:
        print(path)

    raise SystemExit


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading Resume.csv...")
print(resume_file)

df_resume = pd.read_csv(resume_file)

print("Resume.csv shape:", df_resume.shape)


print("\nLoading resume_dataset.csv...")
print(processed_file)

df_processed = pd.read_csv(processed_file)

print("resume_dataset.csv shape:", df_processed.shape)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_resume_columns = [
    "ID",
    "Resume_str",
    "Resume_html",
    "Category"
]

required_processed_columns = [
    "ID",
    "Cleaned_Text",
    "Processed_Text"
]


for column in required_resume_columns:

    if column not in df_resume.columns:

        raise ValueError(
            f"Resume.csv is missing column: {column}"
        )


for column in required_processed_columns:

    if column not in df_processed.columns:

        raise ValueError(
            f"resume_dataset.csv is missing column: {column}"
        )


# ============================================================
# REMOVE DUPLICATES
# ============================================================

df_resume = df_resume.drop_duplicates(
    subset="ID"
)

df_processed = df_processed.drop_duplicates(
    subset="ID"
)


# ============================================================
# COMBINE DATASETS
# ============================================================

print("\nCombining datasets using ID...")


df = pd.merge(
    df_resume[
        [
            "ID",
            "Resume_str",
            "Resume_html",
            "Category"
        ]
    ],

    df_processed[
        [
            "ID",
            "Cleaned_Text",
            "Processed_Text"
        ]
    ],

    on="ID",

    how="inner"
)


# ============================================================
# CLEAN TEXT COLUMNS
# ============================================================

df["Resume_str"] = df["Resume_str"].fillna("")
df["Cleaned_Text"] = df["Cleaned_Text"].fillna("")
df["Processed_Text"] = df["Processed_Text"].fillna("")

df["Category"] = df["Category"].fillna("UNKNOWN")


# ============================================================
# REMOVE EMPTY RESUMES
# ============================================================

df = df[
    df["Resume_str"].str.strip() != ""
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETED")
print("=" * 60)

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal columns:")
print(list(df.columns))

print("\nNumber of categories:")
print(df["Category"].nunique())

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(df.head())

print("\nSUCCESS!")