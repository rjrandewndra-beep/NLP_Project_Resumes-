# Resume Skill Extractor — Member 1

## 1. Member Information

**Member:** Member 1

**Project:** Resume Skill Extractor

**Role:** NLP preprocessing, feature extraction, Logistic Regression, LSTM, candidate ranking, and application foundation.

---

# 2. Member 1 Responsibilities

Member 1 is responsible for the core NLP pipeline and the initial candidate-processing system.

The main responsibilities are:

1. Dataset preparation
2. Text preprocessing
3. Feature extraction using TF-IDF
4. Logistic Regression classification
5. LSTM classification
6. Candidate ranking foundation
7. Resume processing/application foundation
8. Testing and debugging of the above components

---

# 3. Project Pipeline

The Member 1 pipeline is:

```text
Resume Dataset
      |
      v
Dataset Preparation
      |
      v
Text Preprocessing
      |
      v
TF-IDF Feature Extraction
      |
      +-------------------+
      |                   |
      v                   v
Logistic Regression      LSTM
      |                   |
      +---------+---------+
                |
                v
        Candidate Processing
                |
                v
        Candidate Ranking
                |
                v
          Web Application
