
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

DATA_PATH = "data/spam.csv"
MODEL_PATH = "model/spam_model.pkl"


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = {"label", "message"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Dataset must contain 'label' and 'message' columns."
        )

    data = data.dropna(subset=["label", "message"]).copy()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str).str.strip()

    data = data[data["label"].isin(["ham", "spam"])]
    data = data[data["message"] != ""]

    data["label"] = data["label"].map({
        "ham": 0,
        "spam": 1
    })

    if data["label"].nunique() < 2:
        raise ValueError(
            "Dataset must contain both ham and spam messages."
        )

    return data


def train_model(data):
    x = data["message"]
    y = data["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ])

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print("\nModel Evaluation")
    print("-----------------")
    print("Accuracy :", round(accuracy_score(y_test, predictions), 2))
    print("Precision:", round(
        precision_score(y_test, predictions, zero_division=0), 2
    ))
    print("Recall   :", round(
        recall_score(y_test, predictions, zero_division=0), 2
    ))
    print("F1 Score :", round(
        f1_score(y_test, predictions, zero_division=0), 2
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return model


def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("\nModel saved successfully at:", MODEL_PATH)


def main():
    try:
        data = load_data()
        print("Dataset loaded successfully!")
        print("Total messages:", len(data))

        model = train_model(data)
        save_model(model)

        print("\nModel trained successfully!")

    except (FileNotFoundError, ValueError, OSError) as error:
        print("\nError:", error)


if __name__ == "__main__":
    main()
