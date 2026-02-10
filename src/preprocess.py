import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DATA_PATH = Path("data/processed/processed_churn.csv")

def preprocess():
    df = pd.read_csv(RAW_DATA_PATH)

    # Drop customerID (not useful for ML)
    df = df.drop(columns=["customerID"])

    # Convert target variable to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Handle TotalCharges (some values are spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("✅ Data preprocessing completed")

if __name__ == "__main__":
    preprocess()
