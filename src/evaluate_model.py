import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score
from extract_text import extract_text_from_pdf

def evaluate_model(processed_dir="data/processed", model_path="models/doc_classifier.joblib"):
    model = joblib.load(model_path)
    X = []
    y_true = []
    y_pred = []

    processed_dir = Path(processed_dir)

    for class_dir in processed_dir.iterdir():
        if class_dir.is_dir():
            true_label = class_dir.name
            for txt_file in class_dir.glob("*.txt"):
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read()

                prediction = model.predict([text])[0]

                y_true.append(true_label)
                y_pred.append(prediction)

    # Calcul de la précision
    accuracy = accuracy_score(y_true, y_pred)
    correct = sum([yt == yp for yt, yp in zip(y_true, y_pred)])

    print(f"\n✅ Précision manuelle : {correct} / {len(y_true)} = {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model()
