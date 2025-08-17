from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from ingestion.chunker import split_text
from ingestion.document_parser import load_html, load_plain_text


def test_split_text_basic():
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10
    chunks = split_text(text, chunk_size=50, chunk_overlap=0)
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert sum(len(c) for c in chunks) >= len(text) - 50  # allow for final partial chunk


def test_load_plain_text(tmp_path: Path):
    file = tmp_path / "sample.txt"
    file.write_text("line1\n\n line2  \n")
    assert load_plain_text(file) == "line1\nline2"


def test_load_html(tmp_path: Path):
    file = tmp_path / "sample.html"
    file.write_text("<html><body><p>Hello</p><div>World</div></body></html>")
    assert load_html(file) == "Hello\nWorld"
