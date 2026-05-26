class PDFExtractor:
    def __init__(self, file_path, password=None):
        self.file_path = file_path
        self.password = password
        self.doc = fitz.Document(file_path)
        if self.doc.is_encrypted:
            if self.password:
                self.doc.authenticate(self.password)
            else:
                raise ExtractionError("PDF is password protected. Pass password= to extract()")
