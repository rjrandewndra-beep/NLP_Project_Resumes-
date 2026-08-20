# Intelligent Resume Matcher & Skill Extractor Using NLP and Machine Learning

An end-to-end Machine Learning and Deep Learning system designed to automate resume domain categorization, technical skill gap analysis, and dual-weighted candidate ranking against job specifications.

---

## 👥 Team Contributions & Architecture

| Member | Branch | Models Implemented | Role & Responsibilities |
| :--- | :--- | :--- | :--- |
| **Member 1 (Randew)** | `member-1` | Logistic Regression, LSTM | Data preprocessing, Baseline linear classification, Recurrent Neural Net |
| **Member 2 (Janalanka)** | `member-2` | Linear SVM, Bidirectional GRU | High-dimensional TF-IDF SVM, Bi-GRU sequence modeling, Streamlit UI & Final Ranking Engine |
| **Member 3 (Amashanki)** | `member-3` | Random Forest, BERT Transformer | Ensemble tree modeling, Deep Transformer fine-tuning |

---

## 🏆 Multi-Model Benchmark Leaderboard

Evaluated across **2,481 resume records** spanning **24 industry categories**:

| Model Architecture | Category | Accuracy (%) | Macro F1-Score | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | Classical ML (TF-IDF) | **76.50%** | **0.740** | **Best Classical** |
| **BERT Transformer** | Deep Transformer | **75.00%** | **0.730** | **Best Deep Model** |
| **Bidirectional GRU (Bi-GRU)** | Recurrent Neural Net | **72.00%** | **0.700** | High Efficiency |
| **Random Forest** | Ensemble Trees | **68.50%** | **0.660** | Baseline Ensemble |
| **LSTM Network** | Recurrent Neural Net | **65.00%** | **0.630** | Sequential Baseline |
| **Logistic Regression** | Linear Model | **62.00%** | **0.600** | Linear Baseline |

---

## ⚙️ Core Technical Pipeline

1. **Document Intake & Normalization:** PDF/TXT text extraction via `pypdf` with Regex URL, email, and noise filtering.
2. **Domain Classification:** Multi-model inference across 24 industry categories.
3. **Semantic Scoring:** TF-IDF Cosine Similarity ($S_{cos}$) between Job Description and Resume vector spaces.
4. **Competency Gap Analysis:** Domain vocabulary matching ($JD \cap Resume$) and missing skill identification ($JD \setminus Resume$).
5. **Composite Scoring Formula:**
   $$\text{Final Score (\%)} = \left[ 0.60 \times S_{cos} + 0.40 \times \left( \frac{|\text{Matched Skills}|}{|\text{Required Skills}|} \right) \right] \times 100$$

---

## 🚀 Windows Setup & Installation

Open Command Prompt, PowerShell, or the VS Code terminal:

```powershell
# 1. Clone the repository
git clone [https://github.com/rjrandewndra-beep/NLP_Project_Resumes-.git](https://github.com/rjrandewndra-beep/NLP_Project_Resumes-.git)
cd NLP_Project_Resumes-

# 2. Create and activate virtual environment
py -m venv .venv
.venv\Scripts\activate

# 3. Upgrade pip and install dependencies
py -m pip install --upgrade pip
pip install -r requirements_member2.txt

# 4. Launch Streamlit Production Dashboard
python -m streamlit run src/app.py
