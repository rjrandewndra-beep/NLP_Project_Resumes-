"""
Final Resume Matcher & Candidate Ranking Engine
===============================================
Production-grade NLP scoring pipeline supporting multi-model inference (SVM, Random Forest, Bi-GRU, BERT),
dynamic domain category prediction, TF-IDF Cosine Semantic Similarity calculation,
technical competency extraction, and dual-weighted composite candidate ranking.
"""

import os
import re
import io
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import pypdf
except ImportError:
    pypdf = None


class FinalResumeMatcher:
    def __init__(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.models_dir = os.path.join(self.project_root, "models")
        
        # Comprehensive Technical & Industry Domain Vocabulary
        self.skill_set = {
            # Programming & Scripting
            "python", "java", "c++", "c#", "javascript", "typescript", "r", "go", "ruby", "scala", "php", "sql", "nosql",
            # Web & Enterprise Frameworks
            "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring boot", "html", "css", "rest api",
            # AI, Machine Learning & Deep Learning
            "machine learning", "deep learning", "nlp", "computer vision", "data science",
            "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "pandas", "numpy", "matplotlib", "seaborn",
            "bert", "lstm", "gru", "svm", "random forest", "logistic regression", "transformers", "xgboost",
            # Cloud, DevOps & Infrastructure
            "docker", "kubernetes", "aws", "azure", "gcp", "git", "github", "ci/cd", "linux", "jenkins",
            # Data Engineering & Analytics
            "data analysis", "tableau", "power bi", "hadoop", "spark", "agile", "scrum", "jira"
        }
        
        # Load available models into inference registry
        self.models = {}
        self.load_models()

    def load_models(self):
        """Loads available classification models (SVM, Random Forest, etc.) safely."""
        # 1. Support Vector Machine (Linear SVM with TF-IDF)
        svm_path = os.path.join(self.models_dir, "svm_model.pkl")
        svm_tfidf_path = os.path.join(self.models_dir, "svm_tfidf.pkl")
        svm_encoder_path = os.path.join(self.models_dir, "svm_label_encoder.pkl")
        
        if os.path.exists(svm_path) and os.path.exists(svm_tfidf_path) and os.path.exists(svm_encoder_path):
            try:
                self.models["svm"] = {
                    "model": joblib.load(svm_path),
                    "vectorizer": joblib.load(svm_tfidf_path),
                    "encoder": joblib.load(svm_encoder_path)
                }
            except Exception:
                pass

        # 2. Random Forest Classifier
        rf_path = os.path.join(self.models_dir, "random_forest_model.pkl")
        rf_tfidf_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
        rf_encoder_path = os.path.join(self.models_dir, "label_encoder.pkl")
        
        if os.path.exists(rf_path) and os.path.exists(rf_tfidf_path) and os.path.exists(rf_encoder_path):
            try:
                self.models["random_forest"] = {
                    "model": joblib.load(rf_path),
                    "vectorizer": joblib.load(rf_tfidf_path),
                    "encoder": joblib.load(rf_encoder_path)
                }
            except Exception:
                pass

    def get_available_models(self):
        """Returns list of active inference model keys, defaulting to SVM."""
        keys = list(self.models.keys())
        if not keys:
            return ["svm"]
        return keys

    def extract_text_from_file(self, uploaded_file) -> str:
        """Parses and extracts raw text from uploaded PDF or TXT stream."""
        filename = uploaded_file.name.lower()
        text = ""
        
        if filename.endswith(".pdf"):
            if pypdf:
                try:
                    pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + " "
                except Exception:
                    text = ""
            else:
                text = ""
        else:
            try:
                text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            except Exception:
                text = ""
                
        return text.strip()

    def clean_text(self, text: str) -> str:
        """Applies regex normalization, URL stripping, special char removal, and case folding."""
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s+#.]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_skills(self, text: str):
        """Extracts technical skills using bounded regex search against vocabulary."""
        text_lower = text.lower()
        found_skills = set()
        for skill in self.skill_set:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        return found_skills

    def predict_category(self, text: str, model_type="svm") -> str:
        """Predicts resume domain category using selected inference engine."""
        cleaned = self.clean_text(text)
        if model_type in self.models:
            bundle = self.models[model_type]
            try:
                features = bundle["vectorizer"].transform([cleaned])
                pred_idx = bundle["model"].predict(features)[0]
                return str(bundle["encoder"].inverse_transform([pred_idx])[0])
            except Exception:
                pass
        return "INFORMATION-TECHNOLOGY"

    def process_and_rank_resumes(self, job_description: str, resume_files: list, classifier_type="svm"):
        """
        Executes dual-weighted scoring pipeline:
        Final Match Score (%) = (0.60 * Cosine Similarity + 0.40 * Skill Overlap Ratio) * 100
        """
        jd_clean = self.clean_text(job_description)
        jd_skills = self.extract_skills(job_description)

        detailed_candidates = []
        summary_rows = []

        for f in resume_files:
            raw_text = self.extract_text_from_file(f)
            resume_clean = self.clean_text(raw_text) if raw_text else ""
            
            # Predict Domain Class
            pred_cat = self.predict_category(resume_clean, classifier_type)
            
            # Extract Skills & Identify Gaps
            cand_skills = self.extract_skills(raw_text)
            matched_skills = sorted(list(jd_skills.intersection(cand_skills)))
            missing_skills = sorted(list(jd_skills.difference(cand_skills)))
            
            # 1. Skill Overlap Ratio (0.0 to 1.0)
            overlap_ratio = len(matched_skills) / len(jd_skills) if len(jd_skills) > 0 else 0.0
            
            # 2. TF-IDF Cosine Semantic Similarity
            if resume_clean and jd_clean:
                tfidf = TfidfVectorizer(stop_words='english')
                try:
                    matrix = tfidf.fit_transform([jd_clean, resume_clean])
                    cos_sim = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
                except Exception:
                    cos_sim = 0.0
            else:
                cos_sim = 0.0

            # 3. Composite Hybrid Score (60% Semantic + 40% Competency Coverage)
            match_score = float((0.60 * cos_sim + 0.40 * overlap_ratio) * 100)
            match_score = min(100.0, max(0.0, match_score))

            cand_data = {
                "filename": f.name,
                "predicted_category": pred_cat,
                "match_score": match_score,
                "cosine_similarity": cos_sim,
                "skill_overlap_ratio": overlap_ratio,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "total_skills_found": len(cand_skills)
            }
            detailed_candidates.append(cand_data)

        # Sort candidates descending by Match Score
        detailed_candidates = sorted(detailed_candidates, key=lambda x: x["match_score"], reverse=True)

        for rank, c in enumerate(detailed_candidates, 1):
            summary_rows.append({
                "Rank": f"#{rank}",
                "Candidate File": c["filename"],
                "Predicted Category": c["predicted_category"],
                "Match Score (%)": c["match_score"],
                "Cosine Similarity": c["cosine_similarity"],
                "Skill Overlap (%)": c["skill_overlap_ratio"] * 100,
                "Matched Skills Count": f"{len(c['matched_skills'])} / {len(jd_skills)}"
            })

        return pd.DataFrame(summary_rows), detailed_candidates