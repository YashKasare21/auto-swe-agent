import fitz

def extract_structured(file_path, password=None):
    doc = fitz.Document(file_path)
    if doc.is_encrypted:
        if password is None:
            raise ExtractionError("PDF is password protected. Pass password= to extract()")
        if not doc.authenticate(password):
            raise ExtractionError("Incorrect password for PDF.")
    # rest of the function implementation...
