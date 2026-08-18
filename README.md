# Intelligent Resume Skill Extractor Using NLP and Machine Learning

Member 3 implementation for application development, Random Forest, BERT,
repository preparation, and deployment preparation.

## Implemented

- TF-IDF + Random Forest classifier for 24 resume categories
- Accuracy, weighted precision, recall, F1, and full classification report
- CPU-friendly BERT (`prajjwal1/bert-tiny`) fine-tuning with PyTorch and Transformers
- Streamlit flow for job description, multiple resume upload, text extraction,
  preprocessing, skill extraction, category prediction, similarity, skill match,
  final score, and top-10 ranking
- Selectable Random Forest or BERT category prediction in the application
- PDF, DOCX, and TXT resume support

## Windows setup

Open Command Prompt or the VS Code terminal:

```powershell
cd "C:\Users\DVM\OneDrive\Desktop\01 NLP 27 FINAL"
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train Random Forest

```powershell
python src\train_random_forest.py
```

Created files:

- `models/random_forest_model.pkl`
- `models/tfidf_vectorizer.pkl`
- `models/label_encoder.pkl`
- `models/random_forest_metrics.json`
- `models/random_forest_classification_report.csv`

## Train BERT on CPU

The default is a CPU demonstration run using 600 stratified resumes, five
epochs, 128 tokens, and batch size 8. It remains a real BERT classifier with all
24 labels.

```powershell
python src\train_bert.py
```

To train on the full dataset later:

```powershell
python src\train_bert.py --max-samples 0 --epochs 3
```

BERT output is saved in `models/bert_resume_classifier/`, including model,
tokenizer, label mapping, metrics, and classification report.

## Verified model results

The supplied artifacts were evaluated using stratified train/test splits with
random seed 42.

| Model | Training setup | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Random Forest | 1,986 train / 497 test | 75.86% | 78.70% | 75.86% | 74.62% |
| BERT tiny CPU demo | 480 train / 120 test, 5 epochs | 47.50% | 44.92% | 47.50% | 42.00% |

The BERT result is a CPU demonstration baseline, not a claim of production
quality. Train on all 2,484 resumes for a stronger final experiment.

## Run the application

Train Random Forest first, then:

```powershell
streamlit run src/app.py
```

## Ranking formula

The integration uses `Final Score = 60% similarity + 40% skill match`. If the
group's existing ranking module uses different weights, keep that module and
connect its output to the displayed seven columns instead of replacing it.

## Git commands

```powershell
git status
git add .gitignore README.md requirements.txt src models
git commit -m "Member 3: add Random Forest, BERT and Streamlit integration"
git push origin main
```

Review `git status` before committing. Add the dataset only if your team has
agreed to store it in Git and the repository size remains acceptable.

## Deployment status

This repository is prepared for Streamlit deployment, but deployment is not
complete until the application starts successfully in the target environment
and the deployed URL is tested.
