from svm_predict import predict_resume as svm_predict
from gru_predict import predict_resume as gru_predict

SAMPLE_RESUME = """
Software developer with experience in Python, SQL, machine learning,
pandas, scikit-learn, data analysis, APIs and software development.
"""

def main():
    print("=" * 65)
    print("MEMBER 2 MODEL SMOKE TEST")
    print("=" * 65)
    print("\nSVM prediction:")
    print(svm_predict(SAMPLE_RESUME))
    print("\nGRU prediction:")
    print(gru_predict(SAMPLE_RESUME))
    print("\nBoth prediction modules executed successfully.")

if __name__ == "__main__":
    main()
