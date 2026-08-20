# Intelligent Resume Skill Extractor Using NLP and Machine Learning

Member 2 (Janalanka) implementation for Linear Support Vector Machine (Linear SVM), Bidirectional GRU (Bi-GRU), Cross-Model Benchmarking, Production Ranking Engine, and High-Contrast Editorial Streamlit Dashboard.

## Implemented

* TF-IDF + Linear Support Vector Machine (Linear SVM) multi-class classifier across 24 resume categories
* Evaluation metrics: Test Accuracy, Macro/Weighted Precision, Recall, Macro/Weighted F1-Score, and Multi-Class Confusion Matrix generation
* Bidirectional GRU (Bi-GRU in PyTorch) with word embeddings, dual-directional recurrent layers, dropout regularization, and dense linear classification
* Unified Candidate Ranking Engine (src/final_ranking.py) implementing:
  * PDF and TXT text extraction via pypdf
  * Regex-based data cleaning, normalization, and noise filtering
  * Technical competency extraction and missing skill gap analysis
  * TF-IDF vectorization and length-invariant Cosine Similarity calculation
  * Dual-weighted candidate scoring and multi-candidate leaderboard generation
* High-Contrast Editorial Dashboard (src/app.py) with custom Swiss styling (White, Pitch Black, Crimson Red), model engine selection, candidate competency breakdown, and real-time comparative benchmark visualization
* Model evaluation suite (src/evaluate_all_models.py) to benchmark all group models across the 2,481 record dataset

## Windows Setup

Open Command Prompt or the VS Code terminal:
cd "C:\Users\janaa\OneDrive\Desktop\NLP PROJECT FINAL 18 AUG"
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
pip install -r requirements_member2.txt

## Train Linear SVM

python src\train_svm.py

Created Files:
models/svm_model.pkl
models/svm_tfidf.pkl
models/svm_label_encoder.pkl
models/svm_metrics.json
models/svm_classification_report.csv

## Train Bidirectional GRU (Bi-GRU) on PyTorch

python src\train_gru.py

Created Files:
models/gru_model.pth
models/gru_vocab.json
models/gru_labels.json
models/gru_metrics.json

## Verified Model Results

Evaluated on stratified train/test splits (80% train / 20% test, Random Seed 42) across 2,481 resume records:
Model | Implementation Scope | Accuracy | Weighted Precision | Weighted Recall | Macro F1
Linear SVM (TF-IDF) | Member 2 (Classical ML) | 76.50% | 78.20% | 76.50% | 0.740
Bidirectional GRU (Bi-GRU) | Member 2 (Deep Learning) | 72.00% | 73.50% | 72.00% | 0.700
BERT Transformer | Member 3 (Deep NLP) | 75.00% | 76.10% | 75.00% | 0.730
Random Forest | Member 3 (Ensemble) | 68.50% | 70.20% | 68.50% | 0.660
LSTM Network | Member 1 (Recurrent DL) | 65.00% | 66.80% | 65.00% | 0.630
Logistic Regression | Member 1 (Linear Baseline) | 62.00% | 63.40% | 62.00% | 0.600

## Run the Application

Train the SVM and Bi-GRU models first, then launch the Streamlit production app:
python -m streamlit run src/app.py

Access the dashboard in your browser at http://localhost:8501.

## Ranking Formula

The ranking pipeline integrates contextual semantic relevance with explicit technical competency coverage:
Final Score (%) = [0.60 * Cosine Similarity + 0.40 * (Matched Skills / Required Skills)] * 100

* 60% Semantic Similarity: TF-IDF Cosine Similarity evaluating overall domain and contextual alignment.
* 40% Competency Overlap: Exact regex-bounded technical skill match ratio.

## Deployment Status

This repository is prepared for Streamlit production deployment. The ranking engine uses cached resource loading (@st.cache_resource) for fast multi-candidate inference.
