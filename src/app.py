import os
import sys
import tempfile
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT RANKING SYSTEM
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
    page_title="AI Resume Skill Extractor",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .result-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📄 AI-Powered Resume Skill Extractor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'NLP and Machine Learning based Candidate Ranking System'
    '</div>',
    unsafe_allow_html=True
)


st.info(
    """
    Upload a job description and multiple candidate resumes.
    The system extracts relevant information, compares candidates
    with the job requirements, calculates matching scores, and
    ranks the candidates.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Information")

    st.write("**NLP:** Text preprocessing")

    st.write("**Feature Extraction:** TF-IDF")

    st.write("**ML Model:** Logistic Regression")

    st.write("**DL Model:** LSTM")

    st.write("**Ranking:** Skill + similarity matching")

    st.divider()

    st.write("### Input Files")

    st.write("Job Description:")
    st.write("PDF or TXT")

    st.write("Resumes:")
    st.write("Multiple PDF or TXT files")


# ============================================================
# JOB DESCRIPTION UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">1. Upload Job Description</div>',
    unsafe_allow_html=True
)

job_file = st.file_uploader(
    "Upload a job description",
    type=["pdf", "txt"],
    accept_multiple_files=False,
    key="job_description"
)


# ============================================================
# RESUME UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">2. Upload Candidate Resumes</div>',
    unsafe_allow_html=True
)

resume_files = st.file_uploader(
    "Upload one or more resumes",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key="resumes"
)


# ============================================================
# SHOW SELECTED FILES
# ============================================================

if job_file is not None:

    st.success(
        f"Job description selected: {job_file.name}"
    )


if resume_files:

    st.success(
        f"{len(resume_files)} resume(s) selected."
    )

    with st.expander("View selected resumes"):

        for resume in resume_files:
            st.write(f"📄 {resume.name}")


# ============================================================
# TEXT EXTRACTION FUNCTIONS
# ============================================================

def extract_text_from_txt(file_path):
    """
    Read text from a TXT file.
    """

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            return file.read()

    except Exception:

        return ""


def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.

    Uses pypdf if available.
    """

    try:

        from pypdf import PdfReader

        reader = PdfReader(file_path)

        text = []

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text.append(page_text)

        return "\n".join(text)

    except Exception as e:

        st.warning(
            f"Could not extract PDF text from "
            f"{Path(file_path).name}: {e}"
        )

        return ""


def extract_uploaded_file_text(uploaded_file, temporary_directory):
    """
    Save an uploaded file temporarily and extract its text.
    """

    file_name = uploaded_file.name

    file_path = Path(temporary_directory) / file_name

    with open(file_path, "wb") as output_file:

        output_file.write(
            uploaded_file.getbuffer()
        )

    extension = Path(file_name).suffix.lower()

    if extension == ".txt":

        text = extract_text_from_txt(file_path)

    elif extension == ".pdf":

        text = extract_text_from_pdf(file_path)

    else:

        text = ""

    return file_path, text


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown(
    '<div class="section-title">3. Analyze Candidates</div>',
    unsafe_allow_html=True
)


analyze_button = st.button(
    "🚀 Analyze Candidates",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAIN PROCESS
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if job_file is None:

        st.error(
            "Please upload a job description first."
        )

        st.stop()


    if not resume_files:

        st.error(
            "Please upload at least one resume."
        )

        st.stop()


    # --------------------------------------------------------
    # TEMPORARY DIRECTORY
    # --------------------------------------------------------

    temporary_directory = tempfile.mkdtemp(
        prefix="resume_ranking_"
    )


    # --------------------------------------------------------
    # SAVE JOB DESCRIPTION
    # --------------------------------------------------------

    with st.spinner(
        "Reading job description..."
    ):

        job_path, job_description = (
            extract_uploaded_file_text(
                job_file,
                temporary_directory
            )
        )


    if not job_description.strip():

        st.error(
            "Could not extract text from the job description."
        )

        st.stop()


    # --------------------------------------------------------
    # SAVE RESUMES
    # --------------------------------------------------------

    resume_paths = []

    resume_texts = {}

    with st.spinner(
        "Reading candidate resumes..."
    ):

        for resume_file in resume_files:

            resume_path, resume_text = (
                extract_uploaded_file_text(
                    resume_file,
                    temporary_directory
                )
            )

            if resume_text.strip():

                resume_paths.append(
                    str(resume_path)
                )

                resume_texts[
                    str(resume_path)
                ] = resume_text


    # --------------------------------------------------------
    # CHECK RESUMES
    # --------------------------------------------------------

    if not resume_paths:

        st.error(
            "No readable resumes were found."
        )

        st.stop()


    st.write(
        f"Successfully read {len(resume_paths)} resume(s)."
    )


    # --------------------------------------------------------
    # SHOW BASIC TEXT INFORMATION
    # --------------------------------------------------------

    with st.expander(
        "🔎 View extracted job description"
    ):

        st.text(
            job_description[:5000]
        )


    # --------------------------------------------------------
    # RUN CANDIDATE RANKING
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '4. AI Candidate Ranking'
        '</div>',
        unsafe_allow_html=True
    )


    with st.spinner(
        "Running AI candidate ranking..."
    ):

        try:

            results = rank_candidates(
                job_description,
                resume_paths
            )

        except Exception as e:

            st.error(
                "Candidate ranking failed."
            )

            st.exception(e)

            st.stop()


    # ========================================================
    # HANDLE DATAFRAME RESULT
    # ========================================================

    if results is None:

        st.warning(
            "No candidates were returned."
        )

        st.stop()


    # Pandas DataFrame support

    try:

        if results.empty:

            st.warning(
                "No candidates were returned."
            )

            st.stop()

    except AttributeError:

        # If ranking returns a list instead of DataFrame

        if len(results) == 0:

            st.warning(
                "No candidates were returned."
            )

            st.stop()


    # ========================================================
    # RESULTS
    # ========================================================

    st.success(
        f"Successfully ranked {len(results)} candidate(s)."
    )


    st.markdown(
        '<div class="section-title">'
        '🏆 Candidate Ranking Results'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # DATAFRAME DISPLAY
    # --------------------------------------------------------

    if hasattr(results, "columns"):

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # TOP CANDIDATE CARDS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '🥇 Top Candidates'
            '</div>',
            unsafe_allow_html=True
        )


        number_to_display = min(
            10,
            len(results)
        )


        for position in range(
            number_to_display
        ):

            candidate = results.iloc[position]


            # Candidate ID

            candidate_id = candidate.get(
                "Candidate ID",
                candidate.get(
                    "ID",
                    "Unknown"
                )
            )


            # Category

            category = candidate.get(
                "Predicted Category",
                candidate.get(
                    "Category",
                    "Unknown"
                )
            )


            # Final score

            final_score = candidate.get(
                "Final Score",
                0
            )


            # Skill match

            skill_match = candidate.get(
                "Skill Match",
                0
            )


            # Similarity

            similarity = candidate.get(
                "Similarity Score",
                0
            )


            # Matched skills

            matched_skills = candidate.get(
                "Matched Skills",
                ""
            )


            # Skills

            skills = candidate.get(
                "Skills",
                ""
            )


            # ------------------------------------------------
            # DISPLAY CARD
            # ------------------------------------------------

            with st.container():

                st.markdown(
                    '<div class="result-box">',
                    unsafe_allow_html=True
                )


                st.subheader(
                    f"#{position + 1} — Candidate {candidate_id}"
                )


                col1, col2, col3, col4 = st.columns(4)


                with col1:

                    st.metric(
                        "Final Score",
                        f"{float(final_score):.2f}"
                    )


                with col2:

                    st.metric(
                        "Skill Match",
                        f"{float(skill_match):.2f}%"
                    )


                with col3:

                    st.metric(
                        "Similarity",
                        f"{float(similarity):.2f}"
                    )


                with col4:

                    st.write("**Category**")

                    st.write(
                        str(category)
                    )


                if matched_skills:

                    st.write(
                        "**Matched Skills:**"
                    )

                    st.write(
                        str(matched_skills)
                    )


                if skills:

                    st.write(
                        "**Candidate Skills:**"
                    )

                    st.write(
                        str(skills)
                    )


                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    if hasattr(results, "to_csv"):

        csv_data = results.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            label="⬇️ Download Ranking Results",
            data=csv_data,
            file_name="candidate_ranking_results.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Intelligent Resume Skill Extractor | "
    "NLP & Machine Learning Project"
)