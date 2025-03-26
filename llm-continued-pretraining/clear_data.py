import json
import re

# Paths for input (raw) and output (cleaned) JSONL files
input_jsonl = "/home/endpoint11/knowledgebase/test/output2.jsonl"   # Change this to the actual path
output_jsonl = "/home/endpoint11/knowledgebase/test/output_clean.jsonl"  # Change this to the actual path

# Define unwanted patterns (characters, whitespace issues, etc.)
UNWANTED_PATTERNS = [
    r"\n",          # Remove newlines
    r"\t",          # Remove tabs
    r"\s{2,}",      # Remove extra spaces
    r"[^\w\s,.!?;:']",  # Remove special characters except punctuation
]

# Define words/phrases to remove from the text
UNWANTED_WORDS = [
    "Confidential", "Draft", "Do not distribute"  # Modify as needed
]


def clean_text(text):
    """
    Cleans extracted text by:
    - Removing unwanted characters and whitespace issues.
    - Removing specific unwanted words or phrases.
    - Ensuring 'end_of_text' is replaced with '<|end_of_text|>'.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return text  # Return as is if the input is not a string

    # Apply regex patterns to remove unwanted characters
    for pattern in UNWANTED_PATTERNS:
        text = re.sub(pattern, " ", text)

    # Remove specific unwanted words/phrases
    for word in UNWANTED_WORDS:
        text = re.sub(rf"\b{word}\b", "", text, flags=re.IGNORECASE)

    # Replace 'end_of_text' with '<|end_of_text|>'
    text = re.sub(r"\s*end_of_text\s*", "<|end_of_text|>", text)

    # Trim extra spaces and return cleaned text
    return text.strip()


def clean_jsonl(input_path, output_path):
    """
    Reads a JSONL file line by line, cleans the text field, and writes the cleaned data to a new file.

    Args:
        input_path (str): Path to the input JSONL file (raw extracted text).
        output_path (str): Path to save the cleaned JSONL file.

    Returns:
        None
    """
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            try:
                # Load JSON data from the line
                data = json.loads(line)

                # Clean the "text" field if present
                if "text" in data:
                    data["text"] = clean_text(data["text"])

                # Write cleaned data to the output file
                json.dump(data, outfile)
                outfile.write("\n")

            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")

    print(f"✅ Cleaning complete! Saved to: {output_path}")


# Run the script to clean the JSONL file
clean_jsonl(input_jsonl, output_jsonl)
