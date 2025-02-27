import fitz  # PyMuPDF

# Open the PDF document
pdf_path = '/home/endpoint11/knowledgebase/etsi-test/gr_SAI002v010101p.pdf'
doc = fitz.open(pdf_path)

# Extract and print the table of contents (TOC)
toc = doc.get_toc()

# Display the TOC
print("Table of Contents:")
for item in toc:
    level, title, page_num = item
    print(f"Level {level}: {title} (Page {page_num})")

# Optional: Extract the text from all pages
print("\nExtracted Text from the PDF:")
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    text = page.get_text("text")
    print(f"\n--- Page {page_num + 1} ---")
    print(text)

# Close the document
doc.close()