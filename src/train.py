import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

DATA_PATH = Path("data/processed/processed_churn.csv")
MODEL_PATH = Path("models/churn_model.pkl")

def train():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"✅ Model trained successfully | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()
