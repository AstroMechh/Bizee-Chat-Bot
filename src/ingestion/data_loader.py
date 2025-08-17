from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from .chunker import split_text
from .document_parser import load_document


def create_vector_db(
    file_path: str | Path = "data/processed_text/california_llc_guide.txt",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    save_dir: str | Path = "data/knowledge_base/faiss_index_california",
) -> FAISS:
    """Create a FAISS vector store from a document.

    Parameters
    ----------
    file_path: str or Path
        Path to the source document. Supported formats include plain text and
        HTML.
    chunk_size: int, optional
        Size of text chunks produced by the splitter. Defaults to ``1000``.
    chunk_overlap: int, optional
        Overlap between consecutive chunks. Defaults to ``150``.
    save_dir: str or Path, optional
        Directory where the vector store will be saved.
    """

    # Load and chunk the document
    text = load_document(file_path)
    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Initialize the embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # Create the FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Save the vector store
    save_path = Path(save_dir)
    os.makedirs(save_path.parent, exist_ok=True)
    vector_store.save_local(str(save_path))
    return vector_store


if __name__ == "__main__":  # pragma: no cover
    create_vector_db()
