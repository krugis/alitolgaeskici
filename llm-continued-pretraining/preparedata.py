import os
import fitz  # PyMuPDF
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")  # Ensure the tokenizer is available

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + " "
    return text.strip()

def chunk_text(text, chunk_size=512):
    """Split text into chunks of a given size."""
    words = word_tokenize(text)
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def process_pdfs_to_csv(input_folder, output_csv):
    """Read multiple PDFs, process them in 512-word batches, and save to CSV."""
    data = []
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, file_name)
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text, chunk_size=512)
            
            for chunk in chunks:
                data.append([file_name, chunk])

    df = pd.DataFrame(data, columns=["filename", "text"])
    df.to_csv(output_csv, index=False, encoding="utf-8")

# Usage example:
input_folder = "path/to/pdf/folder"
output_csv = "output.csv"
process_pdfs_to_csv(input_folder, output_csv)
