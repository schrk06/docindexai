# src/train_classifier.py

from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from stop_words import get_stop_words
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

french_stop_words = set(get_stop_words('french'))

combined_stop_words = ENGLISH_STOP_WORDS.union(french_stop_words)


def load_dataset(processed_dir="data/processed"):
    texts = []
    labels = []

    for class_dir in Path(processed_dir).iterdir():
        if class_dir.is_dir():
            label = class_dir.name
            for txt_file in class_dir.glob("*.txt"):
                with open(txt_file, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(label)

    return texts, labels

def main():
    print("📂 Chargement des données...")
    X, y = load_dataset()

    if not X:
        print("❌ Aucun fichier trouvé. Exécute d'abord scripts/pdf_to_txt_dataset.py")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 Entraînement du modèle...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=list(combined_stop_words), lowercase=True)),  # Utilise 'english' pour éviter l'erreur
        ('clf', MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)

    print("\n📊 Rapport de classification :")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("\n📦 Sauvegarde du modèle...")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/doc_classifier.joblib")
    print("✅ Modèle sauvegardé dans models/doc_classifier.joblib")

if __name__ == "__main__":
    main()
