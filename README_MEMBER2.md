# Member 2 — SVM + GRU Module

Member: D.M.J.B. Disanayake

Responsibilities:
- ML: SVM
- DL: GRU
- Evaluation and prediction utilities

Expected dataset:
dataset/processed/prepared_resume_dataset.csv

Expected columns:
ID, Resume_str, Resume_html, Category, Cleaned_Text, Processed_Text

Target: Category
Default text: Processed_Text

Run from the project root:
python src/train_svm.py
python src/train_gru.py
python src/test_member2_models.py

The model artifacts are generated locally by the training scripts and are not included in this package. This avoids shipping large/binary files that should be trained from the team's dataset.

Important: SVM and GRU perform resume-category classification. They are components/signals for the downstream candidate-ranking system; they are not by themselves the final ranking algorithm.
