from .extract_text import extract_text_from_pdf
import joblib
import sys
from pathlib import Path

def predict_pdf_class(pdf_path, model_path="models/doc_classifier.joblib"):
    print(f"🔍 Lecture du PDF : {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    print("Chargement du modèle...")
    model = joblib.load(model_path)

    print("Prédiction en cours...")
    prediction = model.predict([text])[0]

    print(f"\n📄 Le document est classé comme : **{prediction.upper()}**")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python src/classify.py chemin/vers/fichier.pdf")
    else:
        predict_pdf_class(sys.argv[1])
