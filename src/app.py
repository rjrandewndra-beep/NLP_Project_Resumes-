"""Streamlit candidate-ranking application. Run: streamlit run src/app.py"""

from __future__ import annotations

import io
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

# Extend this project vocabulary when the team finalizes its skill dictionary.
SKILLS = {
    "python", "java", "javascript", "sql", "r", "c++", "c#", "html", "css",
    "react", "angular", "node.js", "flutter", "django", "flask", "fastapi",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "machine learning", "deep learning", "nlp", "bert", "data analysis",
    "data visualization", "power bi", "tableau", "excel", "spark", "hadoop",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
    "mongodb", "mysql", "postgresql", "firebase", "rest api", "linux",
    "communication", "leadership", "project management", "agile", "scrum",
}


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text.lower())
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.getvalue()
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
    if suffix == ".docx":
        return "\n".join(p.text for p in Document(io.BytesIO(raw)).paragraphs)
    if suffix == ".txt":
        return raw.decode("utf-8", errors="ignore")
    raise ValueError("Supported resume formats are PDF, DOCX and TXT.")


def extract_skills(text: str) -> list[str]:
    normalized = f" {clean_text(text)} "
    found = []
    for skill in SKILLS:
        pattern = rf"(?<![a-z0-9]){re.escape(skill)}(?![a-z0-9])"
        if re.search(pattern, normalized):
            found.append(skill)
    return sorted(found)


@st.cache_resource
def load_classifier():
    paths = {
        "model": MODEL_DIR / "random_forest_model.pkl",
        "vectorizer": MODEL_DIR / "tfidf_vectorizer.pkl",
        "encoder": MODEL_DIR / "label_encoder.pkl",
    }
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Train Random Forest first. Missing: " + ", ".join(missing))
    return tuple(joblib.load(paths[key]) for key in ("model", "vectorizer", "encoder"))


@st.cache_resource
def load_bert_classifier():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install torch and transformers before using BERT.") from exc
    path = MODEL_DIR / "bert_resume_classifier"
    if not (path / "config.json").exists():
        raise FileNotFoundError("Train BERT first with: python src/train_bert.py")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return model, tokenizer, torch


def predict_categories(texts: list[str], classifier_name: str) -> list[str]:
    if classifier_name == "BERT":
        model, tokenizer, torch = load_bert_classifier()
        predictions = []
        for start in range(0, len(texts), 8):
            encoded = tokenizer(
                texts[start : start + 8],
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            with torch.no_grad():
                ids = model(**encoded).logits.argmax(dim=1).tolist()
            predictions.extend(model.config.id2label[int(index)] for index in ids)
        return predictions
    model, vectorizer, encoder = load_classifier()
    return encoder.inverse_transform(model.predict(vectorizer.transform(texts))).tolist()


def rank_candidates(
    job_description: str, candidates: list[dict], classifier_name: str = "Random Forest"
) -> pd.DataFrame:
    """Return the existing required ranking columns without changing their meaning."""
    jd_clean = clean_text(job_description)
    resume_texts = [candidate["cleaned_text"] for candidate in candidates]
    match_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = match_vectorizer.fit_transform([jd_clean] + resume_texts)
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten() * 100
    required_skills = set(extract_skills(jd_clean))

    categories = predict_categories(resume_texts, classifier_name)
    rows = []
    for candidate, category, similarity in zip(candidates, categories, similarities):
        skills = set(extract_skills(candidate["cleaned_text"]))
        matched = required_skills.intersection(skills)
        skill_match = (len(matched) / len(required_skills) * 100) if required_skills else 0.0
        final_score = 0.60 * float(similarity) + 0.40 * skill_match
        rows.append(
            {
                "Candidate ID": candidate["candidate_id"],
                "Predicted Category": category,
                "Similarity Score": round(float(similarity), 2),
                "Skill Match": round(skill_match, 2),
                "Final Score": round(final_score, 2),
                "Matched Skills": ", ".join(sorted(matched)) or "None",
                "Skills": ", ".join(sorted(skills)) or "None",
            }
        )
    return pd.DataFrame(rows).sort_values("Final Score", ascending=False).head(10).reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="Intelligent Resume Skill Extractor", layout="wide")
    st.title("Intelligent Resume Skill Extractor")
    st.write("Upload resumes, compare them with a job description, and display the top 10 candidates.")
    job_description = st.text_area("Job description", height=180)
    uploaded_files = st.file_uploader(
        "Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )
    classifier_name = st.selectbox("Category classifier", ["Random Forest", "BERT"])
    if st.button("Rank candidates", type="primary"):
        if not job_description.strip() or not uploaded_files:
            st.warning("Enter a job description and upload at least one resume.")
            return
        candidates, errors = [], []
        for index, uploaded in enumerate(uploaded_files, start=1):
            try:
                text = extract_text(uploaded)
                if not text.strip():
                    raise ValueError("No readable text found")
                candidates.append(
                    {
                        "candidate_id": Path(uploaded.name).stem or f"Candidate-{index}",
                        "cleaned_text": clean_text(text),
                    }
                )
            except Exception as exc:
                errors.append(f"{uploaded.name}: {exc}")
        if errors:
            st.warning("Some files were skipped:\n" + "\n".join(errors))
        if candidates:
            try:
                results = rank_candidates(job_description, candidates, classifier_name)
            except Exception as exc:
                st.error(str(exc))
                return
            st.subheader("Top 10 candidates")
            st.dataframe(results, use_container_width=True, hide_index=True)
            st.download_button(
                "Download ranking CSV",
                results.to_csv(index=False).encode("utf-8"),
                "top_10_candidates.csv",
                "text/csv",
            )


if __name__ == "__main__":
    main()
