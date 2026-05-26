from docstream.core.extractor_v2 import ExtractionError


def test_extraction_error_is_exception():
    assert issubclass(ExtractionError, Exception)
