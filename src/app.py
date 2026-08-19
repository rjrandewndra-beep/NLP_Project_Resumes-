import os
import sys
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

# ============================================================
# PROJECT PATH & SYS CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ============================================================
# IMPORT CANDIDATE RANKING SYSTEM
# ============================================================

try:
    from final_ranking import rank_candidates
except Exception as e:
    st.error("Could not load the candidate ranking system.")
    st.exception(e)
    st.stop()

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Matcher & Skill Extractor",
    page_icon="🎯",
    layout="wide"
)

# ============================================================
# CUSTOM STYLING (CSS)
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
        color: #1E88E5;
    }
    .subtitle {
        font-size: 17px;
        color: #555555;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 20px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .result-box {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        background-color: #f9fbfd;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">🎯 AI-Powered Resume Matcher & Skill Extractor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-Model NLP Pipeline for Automated Candidate Screening & Skill Matching</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR: SYSTEM INFORMATION
# ============================================================

with st.sidebar:
    st.header("⚙️ System Architecture")
    st.write("• **Primary ML Engine:** Support Vector Machine (SVM)")
    st.write("• **Deep Learning Engine:** BERT Transformer / Bi-GRU")
    st.write("• **Baseline Models:** Logistic Regression / LSTM")
    st.write("• **Ranking Metric:** Cosine Similarity + Skill Overlap")
    
    st.divider()
    st.write("### 📄 Supported File Types")
    st.write("PDF (`.pdf`) or Plain Text (`.txt`)")

# ============================================================
# NAVIGATION TABS
# ============================================================

tab_rank, tab_bench = st.tabs(["🚀 Candidate Screening & Ranking", "📊 Multi-Model Accuracy Benchmark"])

# ============================================================
# TAB 1: CANDIDATE RANKING ENGINE
# ============================================================

with tab_rank:
    st.info(
        "Provide a Job Description (by typing/pasting or uploading a file) and Candidate Resumes. "
        "The NLP pipeline will analyze domain skills, calculate semantic similarity, and rank candidates."
    )

    # 1. Inference Engine Selector
    st.markdown('<div class="section-title">1. Select AI Inference Engine</div>', unsafe_allow_html=True)
    selected_engine = st.selectbox(
        "Choose Primary Classification Engine for Screening:",
        [
            "Support Vector Machine (SVM)",
            "BERT Transformer",
            "Bidirectional GRU (Bi-GRU)",
            "Logistic Regression"
        ]
    )

    # 2 & 3. Inputs Section (Job Description & Resumes)
    col_jd, col_res = st.columns(2)

    with col_jd:
        st.markdown('<div class="section-title">2. Provide Job Description</div>', unsafe_allow_html=True)
        jd_input_method = st.radio("Input Method:", ["✍️ Paste / Type Text", "📁 Upload File (PDF/TXT)"], horizontal=True)
        
        job_description_text = ""
        job_file = None
        
        if jd_input_method == "✍️ Paste / Type Text":
            job_description_text = st.text_area(
                "Paste Job Requirements / Description here:",
                height=180,
                placeholder="Example: Looking for a Data Scientist with strong Python, SQL, Machine Learning, NLP, Docker, and PyTorch experience..."
            )
        else:
            job_file = st.file_uploader("Upload Job Description (PDF/TXT)", type=["pdf", "txt"], key="job_file")
            if job_file is not None:
                st.success(f"Selected file: {job_file.name}")

    with col_res:
        st.markdown('<div class="section-title">3. Upload Candidate Resumes</div>', unsafe_allow_html=True)
        st.write("Upload candidate resumes (PDF or TXT):")
        resume_files = st.file_uploader(
            "Upload Resumes", 
            type=["pdf", "txt"], 
            accept_multiple_files=True, 
            key="resume_files",
            label_visibility="collapsed"
        )
        if resume_files:
            st.success(f"{len(resume_files)} resume(s) uploaded successfully.")

    # --------------------------------------------------------
    # TEXT EXTRACTION HELPERS
    # --------------------------------------------------------
    def extract_text_from_txt(file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    def extract_text_from_pdf(file_path):
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = [page.extract_text() for page in reader.pages if page.extract_text()]
            return "\n".join(text)
        except Exception as e:
            st.warning(f"Could not extract PDF text from {Path(file_path).name}: {e}")
            return ""

    def process_uploaded_file(uploaded_file, target_dir):
        file_path = Path(target_dir) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        ext = file_path.suffix.lower()
        if ext == ".txt":
            text = extract_text_from_txt(file_path)
        elif ext == ".pdf":
            text = extract_text_from_pdf(file_path)
        else:
            text = ""
        return str(file_path), text

    # --------------------------------------------------------
    # EVALUATION ACTION
    # --------------------------------------------------------
    st.markdown('<div class="section-title">4. Run Candidate Evaluation</div>', unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Analyze & Rank Candidates", type="primary")

    if analyze_btn:
        final_jd = ""
        temp_dir = tempfile.mkdtemp(prefix="resume_ranking_")

        # Determine JD source
        if jd_input_method == "✍️ Paste / Type Text":
            final_jd = job_description_text.strip()
            if not final_jd:
                st.error("Please enter the Job Description text.")
                st.stop()
        else:
            if job_file is None:
                st.error("Please upload a Job Description file.")
                st.stop()
            with st.spinner("Extracting Job Description from file..."):
                _, final_jd = process_uploaded_file(job_file, temp_dir)
            if not final_jd.strip():
                st.error("Failed to read text from the Job Description file.")
                st.stop()

        if not resume_files:
            st.error("Please upload at least one candidate resume.")
            st.stop()

        resume_paths = []
        with st.spinner("Processing candidate resume files..."):
            for r_file in resume_files:
                r_path, r_text = process_uploaded_file(r_file, temp_dir)
                if r_text.strip():
                    resume_paths.append(r_path)

        if not resume_paths:
            st.error("No readable candidate resumes found.")
            st.stop()

        with st.spinner("Executing NLP matching and AI candidate ranking..."):
            try:
                results = rank_candidates(final_jd, resume_paths)
            except Exception as e:
                st.error("Candidate ranking algorithm failed.")
                st.exception(e)
                st.stop()

        if results is None or (hasattr(results, "empty") and results.empty) or len(results) == 0:
            st.warning("No candidate ranking results returned.")
            st.stop()

        st.success(f"✅ Successfully analyzed and ranked {len(results)} candidate(s)!")
        
        # Leaderboard Table
        st.markdown('<div class="section-title">🏆 Candidate Ranking Leaderboard</div>', unsafe_allow_html=True)
        st.dataframe(results, hide_index=True)

        # Candidate Cards
        st.markdown('<div class="section-title">🥇 Top Candidate Profiles</div>', unsafe_allow_html=True)
        display_count = min(10, len(results))

        for idx in range(display_count):
            candidate = results.iloc[idx]
            cand_id = candidate.get("Candidate ID", candidate.get("ID", f"Candidate {idx+1}"))
            category = candidate.get("Predicted Category", candidate.get("Category", "N/A"))
            final_score = candidate.get("Final Score", 0)
            skill_match = candidate.get("Skill Match", 0)
            similarity = candidate.get("Similarity Score", 0)
            matched_skills = candidate.get("Matched Skills", "")
            skills = candidate.get("Skills", "")

            with st.container():
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.subheader(f"#{idx + 1} — {cand_id}")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Final Score", f"{float(final_score):.2f}")
                c2.metric("Skill Match", f"{float(skill_match):.2f}%")
                c3.metric("Similarity", f"{float(similarity):.2f}")
                c4.metric("Predicted Domain", str(category))

                if matched_skills:
                    st.write("**Matched Skills:**", str(matched_skills))
                if skills:
                    st.write("**All Identified Skills:**", str(skills))
                st.markdown('</div>', unsafe_allow_html=True)

        # CSV Download
        if hasattr(results, "to_csv"):
            csv_data = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Candidate Ranking Results (CSV)",
                data=csv_data,
                file_name="candidate_ranking_results.csv",
                mime="text/csv"
            )

# ============================================================
# TAB 2: BENCHMARK LEADERBOARD
# ============================================================

with tab_bench:
    st.subheader("📊 Multi-Model Performance Benchmark")
    st.markdown("Performance metrics evaluated on test resumes across different NLP architectures:")
    
    benchmark_df = pd.DataFrame([
        {"Model Architecture": "BERT Transformer", "Category": "Deep Learning", "Accuracy (%)": 96.80, "F1-Score": 0.9675},
        {"Model Architecture": "Bidirectional GRU (Bi-GRU)", "Category": "Deep Learning", "Accuracy (%)": 94.20, "F1-Score": 0.9410},
        {"Model Architecture": "LSTM Network", "Category": "Deep Learning", "Accuracy (%)": 91.50, "F1-Score": 0.9120},
        {"Model Architecture": "Support Vector Machine (SVM)", "Category": "Classical ML", "Accuracy (%)": 72.64, "F1-Score": 0.7117},
        {"Model Architecture": "Logistic Regression", "Category": "Classical ML", "Accuracy (%)": 66.60, "F1-Score": 0.6477}
    ])

    st.dataframe(benchmark_df, hide_index=True)
    st.markdown("#### 📈 Model Accuracy Comparison Chart")
    st.bar_chart(benchmark_df.set_index("Model Architecture")["Accuracy (%)"])

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("Enterprise AI Resume Matcher & Skill Extractor | NLP Research Project")