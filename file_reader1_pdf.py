
import os
from pathlib import Path
import PyPDF2

# Get current working directory
cwd = os.getcwd()

# Define PDF file path
file_path = Path(cwd) / "Aarya_Resume.pdf"

# Check if file exists
if file_path.exists():
    with open(file_path, "rb") as pdf_file:   # open in binary mode
        reader = PyPDF2.PdfReader(pdf_file)
        
        print("PDF Content:\n")
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            print(f"\n--- Page {page_num} ---\n")
            print(text)
else:
    print(f"File not found at: {file_path}")
