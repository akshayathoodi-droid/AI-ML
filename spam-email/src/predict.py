
import os
import joblib

MODEL_PATH = "model/spam_model.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Error: Trained model not found.")
        print("Please run train.py first to create the model.")
        return None

    try:
        return joblib.load(MODEL_PATH)
    except (OSError, ValueError, EOFError) as error:
        print("Error: Unable to load the trained model.")
        print("Details:", error)
        return None


def predict_email(model, message):
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]

    confidence = probabilities[int(prediction)] * 100

    if prediction == 1:
        print("\nPrediction: SPAM")
    else:
        print("\nPrediction: NOT SPAM")

    print("Confidence:", round(confidence, 2), "%")


def main():
    print("Spam Email Classifier")
    print("---------------------")

    model = load_model()

    if model is None:
        return

    message = input("Enter an email or message: ").strip()

    if not message:
        print("Please enter a message.")
    elif len(message) < 3:
        print("Message is too short. Please enter at least 3 characters.")
    else:
        try:
            predict_email(model, message)
        except (ValueError, AttributeError) as error:
            print("Error: Unable to classify this message.")
            print("Details:", error)


if __name__ == "__main__":
    main()
