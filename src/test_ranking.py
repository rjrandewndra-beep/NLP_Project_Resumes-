import pandas as pd

from ranking import rank_candidates


df = pd.read_csv(
    "dataset/resume_dataset.csv"
)


job_description = """
We are looking for a software engineer
with Python, machine learning, data science,
SQL, programming and analytical skills.
"""


top_candidates = rank_candidates(
    job_description,
    df
)


print("\n")
print("=" * 60)
print("TOP 10 CANDIDATES")
print("=" * 60)


for candidate in top_candidates:

    print(
        candidate["Candidate"],
        "|",
        candidate["Category"],
        "|",
        candidate["Matching Score"],
        "%"
    )