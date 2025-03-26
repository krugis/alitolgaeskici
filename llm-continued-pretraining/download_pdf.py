import pandas as pd
import requests
import os

# Define constants
CSV_FILE = "/home/endpoint11/knowledgebase/tools/ETSICatalog.csv"  # Path to the CSV file
DOWNLOAD_DIR = "/home/endpoint11/knowledgebase/etsi-test"  # Directory to save downloaded files
N = 250  # Number of documents to download
L = 320  # Starting row (zero-based index)

# Ensure the download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_pdfs(csv_file: str, download_dir: str, start_row: int, num_docs: int):
    """
    Downloads PDF files from a CSV file containing URLs.

    Args:
        csv_file (str): Path to the CSV file containing PDF links.
        download_dir (str): Directory where downloaded files will be saved.
        start_row (int): The starting row index in the CSV file.
        num_docs (int): The number of documents to download.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file, delimiter="\t", encoding="utf-8")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Ensure the starting row is within the valid range
    if start_row >= len(df):
        print("Error: start_row is greater than the number of rows in the CSV file.")
        return

    # Loop through the specified range of PDF links
    for i, url in enumerate(df["PDF link"][start_row:start_row + num_docs], start=start_row):
        try:
            # Validate the URL
            if not isinstance(url, str) or not url.startswith("http"):
                print(f"Skipping invalid URL at row {i}: {url}")
                continue

            # Extract filename from URL
            file_name = os.path.join(download_dir, url.split("/")[-1])

            # Check if the file already exists
            if os.path.exists(file_name):
                print(f"Skipping (already downloaded): {file_name}")
                continue

            print(f"Downloading {url}...")

            # Download the file
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(file_name, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                print(f"Saved: {file_name}")
            else:
                print(f"Failed to download: {url}")

        except Exception as e:
            print(f"Error processing URL {url}: {e}")

# Execute the function
download_pdfs(CSV_FILE, DOWNLOAD_DIR, L, N)
