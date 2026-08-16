import os
import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "logistic_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR, "models", "tfidf.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR, "models", "label_encoder.pkl"
)


# Load trained machine learning components
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
encoder = joblib.load(ENCODER_PATH)


def rank_candidates(job_description, resume_dataframe):

    """
    Rank multiple resumes against a job description.

    Parameters:
        job_description: text describing the job
        resume_dataframe: dataframe containing resumes

    Returns:
        Top 10 ranked candidates
    """

    texts = resume_dataframe["Resume_str"].fillna("").astype(str).tolist()

    documents = [job_description] + texts

    vectors = vectorizer.transform(documents)

    job_vector = vectors[0]
    resume_vectors = vectors[1:]

    similarities = cosine_similarity(
        job_vector,
        resume_vectors
    )[0]

    results = []

    for i, score in enumerate(similarities):

        prediction = model.predict(
            resume_vectors[i]
        )[0]

        category = encoder.inverse_transform(
            [prediction]
        )[0]

        results.append({
            "Candidate": i + 1,
            "Category": category,
            "Matching Score": round(score * 100, 2),
            "Resume": texts[i]
        })

    results.sort(
        key=lambda x: x["Matching Score"],
        reverse=True
    )

    return results[:10]


if __name__ == "__main__":

    print("Candidate Ranking Module Loaded Successfully.")