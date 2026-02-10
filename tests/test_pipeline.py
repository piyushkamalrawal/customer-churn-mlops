import os

def test_processed_data_exists():
    assert os.path.exists("data/processed/processed_churn.csv")

def test_model_exists():
    assert os.path.exists("models/churn_model.pkl")
