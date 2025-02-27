import pandas as pd
import requests
import os

# Path to the CSV file
csv_file = "/home/endpoint11/knowledgebase/tools/ETSICatalog.csv"

# Directory to save downloaded files
download_dir = "/home/endpoint11/knowledgebase/etsi-test"
os.makedirs(download_dir, exist_ok=True)  # Ensure the directory exists

# Number of documents to download
N = 250  # Change this as needed

# Starting row (zero-based index)
L = 320  # Change this to the desired starting row

# Load the CSV file
df = pd.read_csv(csv_file, delimiter="\t", encoding="utf-8")

# Ensure L is within the valid range
if L >= len(df):
    print("Error: L is greater than the number of rows in the CSV file.")
    exit()

# Download from Lth row onward, up to N documents
for i, url in enumerate(df["PDF link"][L:L+N], start=L):
    try:
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
