from src.extract_text import extract_text_from_pdf

if __name__ == "__main__":
    pdf_path = "data/raw/cv/cv1.pdf"  # remplace par le nom de ton fichier
    texte = extract_text_from_pdf(pdf_path)
    print("\n=== Texte extrait ===\n")
    print(texte[:1000])  # on affiche seulement les 1000 premiers caractères
