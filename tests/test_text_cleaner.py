"""Tests for text cleaning utilities."""
from lexar.utils.text_cleaner import clean_text


def test_collapse_whitespace():
    assert clean_text("hello   world") == "hello world"


def test_strip_newlines():
    assert clean_text("line1\n\nline2") == "line1 line2"


def test_strip_leading_trailing():
    assert clean_text("  hello  ") == "hello"


def test_empty_string():
    assert clean_text("") == ""


def test_tabs():
    assert clean_text("col1\tcol2") == "col1 col2"
