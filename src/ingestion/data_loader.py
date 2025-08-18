from __future__ import annotations

import os
from pathlib import Path

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from src.config.settings import AppSettings, to_state_slug
from .chunker import split_text
from .document_parser import load_document


def create_vector_db(
    state: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    save_dir: str | Path | None = None,
) -> FAISS:
    """Create a FAISS vector store from all processed files for ``state``."""

    state_slug = to_state_slug(state)
    processed_dir = Path(AppSettings.PROCESSED_TEXT_DIR)
    files = sorted(processed_dir.glob(f"{state_slug}_*_content.txt"))
    if not files:
        raise FileNotFoundError(f"No processed files found for state: {state_slug}")

    text = "\n\n".join(load_document(p) for p in files)
    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Initialize the embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # Create the FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Save the vector store
    save_path = Path(
        save_dir
        or os.path.join(AppSettings.KNOWLEDGE_BASE_DIR, f"faiss_index_{state_slug}")
    )
    os.makedirs(save_path.parent, exist_ok=True)
    vector_store.save_local(str(save_path))
    return vector_store


if __name__ == "__main__":  # pragma: no cover
    create_vector_db("california")
