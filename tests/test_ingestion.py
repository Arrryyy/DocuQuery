import os
from pathlib import Path
import pytest
from src.ingestion import clean_text, chunk_text, ingest_documents, save_chunks
from langchain.docstore.document import Document

def test_clean_text():
    input_text = "  This is   a Test!   "
    cleaned = clean_text(input_text)
    assert cleaned == "this is a test!"

def test_chunk_text():
    text = "abcdefghij"
    # For example, with chunk_size=4 and chunk_overlap=1:
    chunks = chunk_text(text, chunk_size=4, chunk_overlap=1)
    # We expect at least one chunk, and each chunk should be non-empty.
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert len(chunk) > 0

def test_ingest_and_save_chunks(tmp_path):
    # Create a temporary data directory with one sample text file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_file = data_dir / "sample.txt"
    sample_file.write_text("This is a sample text for testing ingestion.", encoding="utf-8")
    
    # Ingest documents with small chunk sizes for testing purposes.
    docs = ingest_documents(str(data_dir), chunk_size=10, chunk_overlap=2)
    # Ensure that some documents were returned
    assert len(docs) > 0
    # Save chunks to a subfolder
    chunks_dir = data_dir / "chunks"
    save_chunks(docs, str(chunks_dir))
    # Verify that chunk files exist
    chunk_files = list(chunks_dir.glob("*.txt"))
    assert len(chunk_files) > 0