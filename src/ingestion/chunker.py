from __future__ import annotations

from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs,
) -> List[str]:
    """Split text into smaller chunks.

    This utility wraps :class:`~langchain.text_splitter.RecursiveCharacterTextSplitter`
    allowing callers to configure the chunk size and overlap used for splitting
    arbitrary text into smaller pieces.

    Parameters
    ----------
    text: str
        The raw text to split.
    chunk_size: int, optional
        Maximum size (in characters) for each chunk. Defaults to ``1000``.
    chunk_overlap: int, optional
        Number of characters to overlap between consecutive chunks. Defaults to
        ``200``.
    **kwargs:
        Additional keyword arguments forwarded to
        :class:`RecursiveCharacterTextSplitter`.

    Returns
    -------
    List[str]
        A list of text chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs
    )
    return splitter.split_text(text)
