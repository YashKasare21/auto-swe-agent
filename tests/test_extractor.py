import pytest
from docstream.core.extractor import PDFExtractor


def test_password_protected_pdf():
    # Create a mock encrypted PDF scenario
    # Verify the correct ExtractionError is raised when no password is provided
    with pytest.raises(ExtractionError):
        PDFExtractor("protected.pdf")
