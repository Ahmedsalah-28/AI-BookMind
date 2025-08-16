import re
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def clean_text(text):
    """Remove page numbers, headers, and extra spaces"""
    text = re.sub(r"\n?\s*Page\s*\d+\s*\n?", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[^\S\r\n]{2,}", " ", text)  # إزالة المسافات المكررة
    text = re.sub(r"\s+\n", "\n", text)  # إزالة المسافات قبل السطر الجديد

    return text.strip()
