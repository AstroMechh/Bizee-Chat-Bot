from __future__ import annotations

from pathlib import Path
from typing import Callable

from bs4 import BeautifulSoup


def normalize_text(text: str) -> str:
    """Normalize whitespace in a block of text."""
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def load_plain_text(path: str | Path) -> str:
    """Load and normalize text from a plain ``.txt`` file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return normalize_text(text)


def load_html(path: str | Path) -> str:
    """Load an HTML file and return its normalized text content."""
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    text = soup.get_text(separator="\n")
    return normalize_text(text)


def load_document(path: str | Path) -> str:
    """Load a document based on its file extension."""
    path = Path(path)
    loader: Callable[[str | Path], str]
    if path.suffix.lower() in {".txt"}:
        loader = load_plain_text
    elif path.suffix.lower() in {".html", ".htm"}:
        loader = load_html
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return loader(path)
