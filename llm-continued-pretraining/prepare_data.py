import fitz  # PyMuPDF
import json
import os
import spacy
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn

# Example usage
pdf_folder = '/home/endpoint11/knowledgebase/etsi-test'
output_path = '/home/endpoint11/knowledgebase/test/output2.jsonl'
chunk_size = 256

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")
console = Console()

def split_into_chunks(text, chunk_size=256):
    sentences = text.split('.')
    sentences = [sentence.strip() + '.' for sentence in sentences if sentence.strip()]

    chunks = []
    current_chunk = []
    current_chunk_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if current_chunk_word_count + sentence_word_count > chunk_size:
            chunks.append({"text": " ".join(current_chunk) + " <|end_of_text|>"})
            current_chunk = [sentence]
            current_chunk_word_count = sentence_word_count
        else:
            current_chunk.append(sentence)
            current_chunk_word_count += sentence_word_count
    
    if current_chunk:
        chunks.append({"text": " ".join(current_chunk) + " <|end_of_text|>"})
    return chunks

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    content = []
    scope_page = history_page = None

    for entry in toc:
        level, title, page_num = entry
        if "Scope" in title:
            scope_page = page_num
        elif "History" in title:
            history_page = page_num

    if not scope_page or not history_page:
        console.print(f"[bold red]Skipping {pdf_path}: 'Scope' or 'History' not found in TOC.[/bold red]")
        return []
    
    total_words = 0
    total_chunks = 0
    
    for page_num in range(scope_page, history_page - 1):
        page = doc.load_page(page_num)
        raw_text = page.get_text("text")
        cleaned_text = " ".join(raw_text.split("\n")[4:]).strip()
        
        if cleaned_text:
            chunks = split_into_chunks(cleaned_text)
            total_words += len(cleaned_text.split())
            total_chunks += len(chunks)
            content.extend(chunks)
    
    doc.close()
    return content, total_words, total_chunks

def process_pdfs_in_folder(folder_path, output_path, chunk_size=256):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    total_files = len(pdf_files)
    total_words = 0
    total_chunks = 0
    
    with Progress(
        TextColumn("Processing: [bold blue]{task.fields[filename]}[/bold blue]"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Processing PDFs", total=total_files, filename="Starting...")
        
        with open(output_path, 'a') as f:
            for i, pdf_file in enumerate(pdf_files, 1):
                pdf_path = os.path.join(folder_path, pdf_file)
                progress.update(task, filename=pdf_file)
                content, words, chunks = extract_text_from_pdf(pdf_path)
                
                for item in content:
                    json.dump(item, f)
                    f.write("\n")
                
                total_words += words
                total_chunks += chunks
                
                console.print(f"[bold green]Processed:[/bold green] {pdf_file} | [cyan]Words:[/cyan] {total_words} | [magenta]Chunks:[/magenta] {total_chunks}")
                progress.advance(task)
    
    console.print(f"\n[bold white on blue]Completed processing {total_files} PDFs! Total words: {total_words}, Total chunks: {total_chunks}[/bold white on blue]")


process_pdfs_in_folder(pdf_folder, output_path, chunk_size=256)
