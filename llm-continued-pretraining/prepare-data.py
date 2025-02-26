import os
import json
from pathlib import Path
from pypdf import PdfReader

# Set the directory containing PDFs
PDF_FOLDER = "/home/endpoint11/knowledgebase/test"  # Change this to your actual folder path
OUTPUT_FILE = "/home/endpoint11/knowledgebase/test/output.jsonl"
CHUNK_SIZE = 512
EOS_TOKEN = "<|endoftext|>"  # End of text token

def extract_text_from_pdfs(folder_path):
    """Extracts text from all PDFs in the specified folder."""
    all_text = []
    
    for pdf_file in Path(folder_path).glob("*.pdf"):
        reader = PdfReader(str(pdf_file))
        pdf_text = ""
        
        for page in reader.pages:
            pdf_text += page.extract_text() or " "  # Extract text, avoid NoneType
        
        all_text.append(pdf_text.strip())  # Clean extra spaces/newlines
    
    return " ".join(all_text)  # Merge text from all PDFs

def chunk_text(text, chunk_size):
    """Splits text into chunks of a given size."""
    return [text[i:i+chunk_size] + EOS_TOKEN for i in range(0, len(text), chunk_size)]

def save_chunks_to_jsonl(chunks, output_file):
    """Saves the text chunks in JSONL format."""
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump({"text": chunk}, f, ensure_ascii=False)
            f.write("\n")

# Process PDFs
pdf_text = extract_text_from_pdfs(PDF_FOLDER)
text_chunks = chunk_text(pdf_text, CHUNK_SIZE)
save_chunks_to_jsonl(text_chunks, OUTPUT_FILE)

print(f"Processed {len(text_chunks)} chunks and saved to {OUTPUT_FILE}.")
