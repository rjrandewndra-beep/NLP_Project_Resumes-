import os
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# PATHS CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Member 2 Best SVM Model Paths
SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"
SVM_TFIDF_PATH = MODELS_DIR / "svm_tfidf.pkl"
SVM_LE_PATH = MODELS_DIR / "svm_label_encoder.pkl"

# Fallback Paths
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

# Predefined Technical & Soft Skills Catalog
SKILL_PATTERNS = [
    "python", "java", "c++", "c#", "r", "sql", "nosql", "mongodb", "postgresql",
    "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
    "nlp", "bert", "lstm", "gru", "svm", "transformers", "huggingface", "spacy", "nltk",
    "docker", "kubernetes", "git", "github", "aws", "azure", "gcp", "fastapi", "flask",
    "django", "react", "html", "css", "javascript", "machine learning", "deep learning",
    "data science", "data analysis", "data engineering", "communication", "leadership", "agile", "scrum"
]

def clean_text(text: str) -> str:
    """Preprocess and clean raw text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '  ', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def extract_skills_from_text(text: str) -> list:
    """Extract verified skills matching the technical catalog."""
    text_lower = text.lower()
    found = [skill.title() for skill in SKILL_PATTERNS if re.search(r'\b' + re.escape(skill) + r'\b', text_lower)]
    return sorted(list(set(found)))

def predict_domain(text: str) -> str:
    """Predict candidate domain using Member 2 Best SVM (with Fallback)."""
    cleaned = clean_text(text)
    
    # 1. Member 2 Best SVM Engine
    if SVM_MODEL_PATH.exists() and SVM_TFIDF_PATH.exists() and SVM_LE_PATH.exists():
        try:
            svm = joblib.load(SVM_MODEL_PATH)
            vec = joblib.load(SVM_TFIDF_PATH)
            le = joblib.load(SVM_LE_PATH)
            pred_idx = svm.predict(vec.transform([cleaned]))[0]
            return str(le.inverse_transform([pred_idx])[0])
        except Exception:
            pass

    # 2. Fallback Logistic Regression Engine
    if LOGISTIC_MODEL_PATH.exists() and TFIDF_PATH.exists():
        try:
            lr = joblib.load(LOGISTIC_MODEL_PATH)
            vec = joblib.load(TFIDF_PATH)
            pred = lr.predict(vec.transform([cleaned]))[0]
            return str(pred)
        except Exception:
            pass

    return "Information Technology / Data Science"

def rank_candidates(job_description: str, resume_paths: list) -> pd.DataFrame:
    """
    Rank candidate resumes against the job description using:
    1. Skill Match Percentage
    2. Cosine Similarity (TF-IDF)
    3. Category Domain Prediction
    """
    cleaned_jd = clean_text(job_description)
    jd_skills = set(extract_skills_from_text(job_description))

    tfidf = TfidfVectorizer(stop_words='english')
    candidate_records = []
    
    for r_path in resume_paths:
        p = Path(r_path)
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        except Exception:
            raw_text = ""

        cleaned_resume = clean_text(raw_text)
        cand_skills = set(extract_skills_from_text(raw_text))

        # 1. Skill Match Score
        matched = jd_skills.intersection(cand_skills)
        if len(jd_skills) > 0:
            skill_score = (len(matched) / len(jd_skills)) * 100.0
        else:
            skill_score = 50.0 if len(cand_skills) > 0 else 0.0

        # 2. TF-IDF Cosine Similarity Score
        try:
            vectors = tfidf.fit_transform([cleaned_jd, cleaned_resume])
            sim_score = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
        except Exception:
            sim_score = 0.0

        # 3. Predict Job Category / Domain
        predicted_cat = predict_domain(raw_text)

        # 4. Weighted Final Ranking Score
        final_score = (sim_score * 50.0) + (skill_score * 0.5)

        candidate_records.append({
            "Candidate ID": p.name,
            "Final Score": round(final_score, 2),
            "Skill Match": round(skill_score, 2),
            "Similarity Score": round(sim_score, 4),
            "Predicted Category": predicted_cat,
            "Matched Skills": ", ".join(sorted(matched)) if matched else "None",
            "Skills": ", ".join(sorted(cand_skills)) if cand_skills else "None"
        })

    df_results = pd.DataFrame(candidate_records)
    if not df_results.empty:
        df_results = df_results.sort_values(by="Final Score", ascending=False).reset_index(drop=True)

    return df_results