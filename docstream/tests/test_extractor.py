import pytest
from docstream.core.extractor import PDFExtractor

pytest.mark.parametrize("password, expected_error", [
    (None, "PDF is password protected. Pass password= to extract()"),
])
def test_password_protected_pdf(password, expected_error):
    with pytest.raises(ExtractionError, match=expected_error):
        PDFExtractor("protected.pdf", password=password)