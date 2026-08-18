from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "svm_model.pkl"
TFIDF_FILE = MODEL_DIR / "svm_tfidf.pkl"
ENCODER_FILE = MODEL_DIR / "svm_label_encoder.pkl"

def _load():
    for path in (MODEL_FILE, TFIDF_FILE, ENCODER_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required SVM file not found: {path}")
    return joblib.load(MODEL_FILE), joblib.load(TFIDF_FILE), joblib.load(ENCODER_FILE)

def predict_resume(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Resume text must be a non-empty string.")

    model, vectorizer, encoder = _load()
    X = vectorizer.transform([text])
    predicted_id = int(model.predict(X)[0])
    label = encoder.inverse_transform([predicted_id])[0]

    decision = model.decision_function(X)
    if decision.ndim == 1:
        raw_score = float(decision[0])
        confidence = float(1.0 / (1.0 + np.exp(-raw_score)))
    else:
        values = decision[0]
        raw_score = float(values[predicted_id])
        shifted = values - np.max(values)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        confidence = float(probs[predicted_id])

    return {"category": str(label), "score": raw_score, "confidence_score": confidence}

def predict_resumes(texts):
    return [predict_resume(text) for text in texts]

if __name__ == "__main__":
    sample = "Python developer with SQL, machine learning, pandas and data analysis experience."
    print(predict_resume(sample))
