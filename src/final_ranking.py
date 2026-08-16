import os
import re
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_model.pkl"
TFIDF_PATH = MODEL_DIR / "tfidf.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

print("Loading ranking models...")

logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
tfidf_vectorizer = joblib.load(TFIDF_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("Ranking models loaded successfully.")


# ============================================================
# SKILL LIST
# ============================================================

SKILLS = [
    "python",
    "java",
    "javascript",
    "c++",
    "c#",
    "r",
    "sql",
    "mysql",
    "mongodb",
    "postgresql",
    "html",
    "css",
    "react",
    "node",
    "node.js",
    "angular",
    "django",
    "flask",
    "fastapi",
    "tensorflow",
    "pytorch",
    "keras",
    "scikit learn",
    "scikit-learn",
    "machine learning",
    "deep learning",
    "data science",
    "data analysis",
    "data analytics",
    "artificial intelligence",
    "computer vision",
    "nlp",
    "natural language processing",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "tableau",
    "power bi",
    "excel",
    "aws",
    "azure",
    "google cloud",
    "docker",
    "kubernetes",
    "git",
    "github",
    "linux",
    "spark",
    "hadoop",
    "communication",
    "leadership",
    "project management",
    "marketing",
    "finance",
    "accounting",
]


# ============================================================
# TEXT EXTRACTION
# ============================================================

def read_text_file(file_path):
    """
    Read text from TXT or PDF files.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    # --------------------------------------------------------
    # TXT FILE
    # --------------------------------------------------------

    if extension == ".txt":

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            return file.read()

    # --------------------------------------------------------
    # PDF FILE
    # --------------------------------------------------------

    elif extension == ".pdf":

        try:

            from pypdf import PdfReader

            reader = PdfReader(str(file_path))

            text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            return text

        except ImportError:

            raise ImportError(
                "pypdf is not installed. "
                "Run: pip install pypdf"
            )

    else:

        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Use PDF or TXT."
        )


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Basic NLP preprocessing.
    """

    if not text:
        return ""

    text = str(text).lower()

    # Replace common separators
    text = text.replace("/", " ")
    text = text.replace("-", " ")

    # Remove unusual characters
    text = text.replace("ï¼", " ")
    text = text.replace("â", " ")

    # Keep letters and numbers
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text):
    """
    Extract skills from resume/job-description text
    using a predefined skill vocabulary.
    """

    clean_text = preprocess_text(text)

    found_skills = []

    for skill in SKILLS:

        skill_pattern = re.escape(skill.lower())

        pattern = r"(?<!\w)" + skill_pattern + r"(?!\w)"

        if re.search(pattern, clean_text):

            display_skill = skill

            if skill == "scikit-learn":
                display_skill = "scikit-learn"

            elif skill == "scikit learn":
                display_skill = "scikit-learn"

            elif skill == "node.js":
                display_skill = "node.js"

            found_skills.append(display_skill)

    return sorted(set(found_skills))


# ============================================================
# EDUCATION EXTRACTION
# ============================================================

def extract_education(text):
    """
    Simple education extraction for demonstration.
    """

    text_lower = text.lower()

    education_keywords = [
        "bachelor",
        "b.sc",
        "bsc",
        "b.eng",
        "beng",
        "master",
        "m.sc",
        "msc",
        "m.eng",
        "meng",
        "phd",
        "doctorate",
        "degree",
        "diploma",
        "higher diploma",
        "university",
    ]

    found = []

    for keyword in education_keywords:

        if keyword in text_lower:

            found.append(keyword.upper())

    if not found:
        return "Not detected"

    return ", ".join(sorted(set(found)))


# ============================================================
# EXPERIENCE EXTRACTION
# ============================================================

def extract_experience(text):
    """
    Detect approximate years of experience.
    """

    patterns = [
        r"(\d+)\+?\s*years?\s+(?:of\s+)?experience",
        r"(\d+)\+?\s*years?\s+in",
    ]

    matches = []

    for pattern in patterns:

        found = re.findall(
            pattern,
            text.lower()
        )

        matches.extend(found)

    if not matches:
        return "Not detected"

    numbers = []

    for value in matches:

        try:
            numbers.append(int(value))

        except ValueError:
            pass

    if not numbers:
        return "Not detected"

    maximum = max(numbers)

    return f"{maximum}+ years"


# ============================================================
# PREDICT JOB CATEGORY
# ============================================================

def predict_category(text):
    """
    Use Logistic Regression to classify the resume
    into one of the 24 dataset categories.
    """

    clean_text = preprocess_text(text)

    if not clean_text:

        return "Unknown"

    features = tfidf_vectorizer.transform(
        [clean_text]
    )

    prediction = logistic_model.predict(features)

    try:

        category = label_encoder.inverse_transform(
            prediction
        )[0]

    except Exception:

        category = str(prediction[0])

    return category


# ============================================================
# CALCULATE SIMILARITY
# ============================================================

def calculate_similarity(
    job_description,
    resume_text
):
    """
    Calculate TF-IDF cosine similarity between
    the job description and a resume.
    """

    job_clean = preprocess_text(
        job_description
    )

    resume_clean = preprocess_text(
        resume_text
    )

    if not job_clean or not resume_clean:

        return 0.0

    vectors = tfidf_vectorizer.transform(
        [
            job_clean,
            resume_clean
        ]
    )

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return float(similarity)


# ============================================================
# CALCULATE SKILL MATCH
# ============================================================

def calculate_skill_match(
    job_skills,
    resume_skills
):
    """
    Calculate percentage of required job skills
    found in the candidate resume.
    """

    if not job_skills:

        return 0.0, []

    matched_skills = sorted(
        set(job_skills).intersection(
            set(resume_skills)
        )
    )

    percentage = (
        len(matched_skills)
        /
        len(job_skills)
    ) * 100

    return percentage, matched_skills


# ============================================================
# FINAL CANDIDATE SCORE
# ============================================================

def calculate_final_score(
    similarity_score,
    skill_match_score
):
    """
    Combine job-resume similarity and skill matching.

    40% = TF-IDF similarity
    60% = required skill matching
    """

    similarity_percentage = (
        similarity_score * 100
    )

    final_score = (
        similarity_percentage * 0.40
        +
        skill_match_score * 0.60
    )

    return round(
        final_score,
        2
    )


# ============================================================
# MAIN RANKING FUNCTION
# ============================================================

def rank_candidates(
    job_description,
    resume_paths,
    number_of_candidates=10
):
    """
    Rank multiple candidates against a job description.

    Parameters
    ----------
    job_description : str
        Text of the job description.

    resume_paths : list
        List of PDF/TXT resume paths.

    number_of_candidates : int
        Number of candidates to return.

    Returns
    -------
    pandas.DataFrame
        Ranked candidate results.
    """

    print()
    print("=" * 60)
    print("CANDIDATE RANKING")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not job_description:

        raise ValueError(
            "Job description is empty."
        )

    if not resume_paths:

        raise ValueError(
            "No resumes were provided."
        )

    # Make sure number_of_candidates is an integer
    try:

        number_of_candidates = int(
            number_of_candidates
        )

    except Exception:

        number_of_candidates = 10

    # Prevent invalid values
    if number_of_candidates < 1:

        number_of_candidates = 10

    # --------------------------------------------------------
    # Extract skills from job description
    # --------------------------------------------------------

    job_skills = extract_skills(
        job_description
    )

    print()
    print("Required job skills:")

    if job_skills:

        print(
            ", ".join(job_skills)
        )

    else:

        print(
            "No predefined skills detected."
        )

    # --------------------------------------------------------
    # Process candidates
    # --------------------------------------------------------

    results = []

    print()
    print(
        f"Processing {len(resume_paths)} resumes..."
    )

    for resume_path in resume_paths:

        try:

            # Read resume
            resume_text = read_text_file(
                resume_path
            )

            if not resume_text.strip():

                print(
                    f"Skipping empty resume: {resume_path}"
                )

                continue

            # Candidate ID
            candidate_id = Path(
                resume_path
            ).stem

            # Preprocess
            clean_resume = preprocess_text(
                resume_text
            )

            # Extract skills
            resume_skills = extract_skills(
                clean_resume
            )

            # Predict category
            predicted_category = predict_category(
                clean_resume
            )

            # Calculate similarity
            similarity = calculate_similarity(
                job_description,
                clean_resume
            )

            # Calculate skill match
            skill_match, matched_skills = (
                calculate_skill_match(
                    job_skills,
                    resume_skills
                )
            )

            # Final score
            final_score = calculate_final_score(
                similarity,
                skill_match
            )

            # Education
            education = extract_education(
                resume_text
            )

            # Experience
            experience = extract_experience(
                resume_text
            )

            # Store result
            results.append(
                {
                    "Candidate ID": candidate_id,

                    "Predicted Category":
                        predicted_category,

                    "Similarity Score":
                        round(
                            similarity * 100,
                            2
                        ),

                    "Skill Match":
                        round(
                            skill_match,
                            2
                        ),

                    "Final Score":
                        final_score,

                    "Matched Skills":
                        ", ".join(
                            matched_skills
                        ),

                    "Skills":
                        ", ".join(
                            resume_skills
                        ),

                    "Education":
                        education,

                    "Experience":
                        experience,

                    "Resume Path":
                        str(resume_path),
                }
            )

        except Exception as error:

            print()
            print(
                f"Error processing {resume_path}:"
            )

            print(error)

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not results:

        raise ValueError(
            "No resumes could be processed successfully."
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Sort by final score
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        by="Final Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Add ranking
    # --------------------------------------------------------

    results_df.insert(
        0,
        "Rank",
        range(
            1,
            len(results_df) + 1
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Return only requested number of candidates.
    #
    # This must use an INTEGER.
    # --------------------------------------------------------

    results_df = results_df.head(
        number_of_candidates
    ).copy()

    # Reset rank after selecting top candidates
    results_df["Rank"] = range(
        1,
        len(results_df) + 1
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TOP CANDIDATES")
    print("=" * 60)

    display_columns = [
        "Rank",
        "Candidate ID",
        "Predicted Category",
        "Similarity Score",
        "Skill Match",
        "Final Score",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Returning top {len(results_df)} candidates."
    )

    print("=" * 60)

    return results_df


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    print(
        "final_ranking.py loaded successfully."
    )

    print(
        "Use rank_candidates() from app.py "
        "to perform candidate ranking."
    )