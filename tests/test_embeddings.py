from src.embeddings import get_embedding, get_embeddings, load_chunks

def test_get_embedding():
    text = "Hello, world!"
    embedding = get_embedding(text)
    # Check that an embedding is returned as a non-empty list
    assert isinstance(embedding, list)
    assert len(embedding) > 0

def test_get_embeddings():
    texts = ["Hello", "world"]
    embeddings = get_embeddings(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    for emb in embeddings:
        assert isinstance(emb, list)
        assert len(emb) > 0

def test_load_chunks(tmp_path):
    # Create temporary chunk files
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "chunk_1.txt").write_text("Chunk one text.", encoding="utf-8")
    (chunks_dir / "chunk_2.txt").write_text("Chunk two text.", encoding="utf-8")
    
    texts = load_chunks(str(chunks_dir))
    assert isinstance(texts, list)
    assert len(texts) == 2