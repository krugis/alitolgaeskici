import PyPDF2
import csv

def read_pdf_in_batches(pdf_path, batch_size=512):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        num_pages = reader.numPages
        text_batches = []

        for page_num in range(num_pages):
            page = reader.getPage(page_num)
            text = page.extract_text()
            if text:
                for i in range(0, len(text), batch_size):
                    text_batches.append(text[i:i+batch_size])

    return text_batches

def write_batches_to_csv(batches, csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Batch'])
        for batch in batches:
            writer.writerow([batch])

def main():
    pdf_path = 'path/to/your/pdf_file.pdf'
    csv_path = 'path/to/your/output_file.csv'
    batches = read_pdf_in_batches(pdf_path)
    write_batches_to_csv(batches, csv_path)

if __name__ == "__main__":
    main()