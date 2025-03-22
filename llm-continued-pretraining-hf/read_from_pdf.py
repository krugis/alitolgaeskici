import fitz  # PyMuPDF
import os
from pdf2image import convert_from_path
import pytesseract

def extract_text_from_pdf(pdf_path):
    """Extracts text from a scanned PDF using OCR."""
    text = ""
    try:
        images = convert_from_path(pdf_path)  # Convert PDF pages to images
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"  # OCR extraction
    except Exception as e:
        print(f"Error extracting text: {e}")
    
    return text

def save_text_to_file(text, output_path):
    """Saves extracted text to a specified file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"Error saving text file: {e}")

if __name__ == "__main__":
    pdf_file = "/home/trainer011/pdf_output/teknik-sartname-muhurlu-kays-yazilim-hizmeti-26-02-2025-37036517.pdf"  # Change this to your PDF file path
    output_folder = "/home/trainer011/pdf_output"
    output_file = os.path.join(output_folder, "extracted_text.txt")
    
    extracted_text = extract_text_from_pdf(pdf_file)
    save_text_to_file(extracted_text, output_file)
    print(f"Extracted text saved to: {output_file}")
