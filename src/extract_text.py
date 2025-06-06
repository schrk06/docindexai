import pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait le texte brut d'un fichier PDF."""
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except Exception as e:
        print(f"Erreur lors de la lecture du PDF : {e}")
    return full_text.strip()
