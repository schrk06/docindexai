# scripts/pdf_to_txt_dataset.py

import os
from pathlib import Path
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)

def convert_pdfs_to_txt(raw_dir="data/raw", processed_dir="data/processed"):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for category_folder in raw_dir.iterdir():
        if category_folder.is_dir():
            category_name = category_folder.name
            out_category_path = processed_dir / category_name
            out_category_path.mkdir(parents=True, exist_ok=True)

            for pdf_file in category_folder.glob("*.pdf"):
                text = extract_text_from_pdf(pdf_file)
                txt_filename = pdf_file.stem + ".txt"
                with open(out_category_path / txt_filename, "w", encoding="utf-8") as f:
                    f.write(text)

                print(f"✅ {pdf_file.name} converti en texte pour la classe '{category_name}'")

if __name__ == "__main__":
    convert_pdfs_to_txt()
