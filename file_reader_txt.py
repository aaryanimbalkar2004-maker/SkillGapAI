import os
from pathlib import Path

# Get current working directory
cwd = os.getcwd()

# Define file path
file_path = Path(cwd) / "Aarya_Resume2.txt"

# Check if file exists
if file_path.exists():
    with open(file_path, "r") as file:
        content = file.read()
    
    print("Full content:\n")
    print(content)
else:
    print(f"File not found at: {file_path}")
