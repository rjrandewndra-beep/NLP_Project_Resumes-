import os
import pandas as pd


# --------------------------------------------------
# Project location
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# --------------------------------------------------
# Dataset paths
# --------------------------------------------------

dataset_folder = os.path.join(
    BASE_DIR,
    "dataset"
)

resume_dataset_path = os.path.join(
    dataset_folder,
    "resume_dataset.csv"
)

Resume_path = os.path.join(
    dataset_folder,
    "Resume.csv"
)


# --------------------------------------------------
# Function to inspect a dataset
# --------------------------------------------------

def inspect_dataset(file_path, dataset_name):

    print()
    print("=" * 70)
    print(dataset_name)
    print("=" * 70)

    print()
    print("File:")
    print(file_path)

    if not os.path.exists(file_path):

        print()
        print("ERROR: File does not exist.")

        return

    df = pd.read_csv(
        file_path
    )

    print()
    print("Number of rows:")
    print(df.shape[0])

    print()
    print("Number of columns:")
    print(df.shape[1])

    print()
    print("Shape:")
    print(df.shape)

    print()
    print("Column names:")
    print(df.columns.tolist())

    print()
    print("Data types:")
    print(df.dtypes)

    print()
    print("First 5 rows:")
    print(df.head())

    print()
    print("Missing values:")
    print(df.isnull().sum())


# --------------------------------------------------
# Inspect both datasets
# --------------------------------------------------

inspect_dataset(
    resume_dataset_path,
    "DATASET 1 - resume_dataset.csv"
)

inspect_dataset(
    Resume_path,
    "DATASET 2 - Resume.csv"
)