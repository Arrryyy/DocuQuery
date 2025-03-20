import numpy as np
from src.chain import build_prompt, retrieve_top_k

def test_build_prompt():
    query = "What is the capital of France?"
    retrieved_chunks = [
        "Paris is the capital of France.",
        "It is known for its art and culture.",
        "The city is situated on the Seine."
    ]
    prompt = build_prompt(query, retrieved_chunks)
    # Check that the prompt includes both the context and the question
    assert "Context:" in prompt
    assert "Question: What is the capital of France?" in prompt
    assert "Answer:" in prompt

def test_retrieve_top_k():
    # Create dummy embeddings (5 chunks, each with 3 features)
    chunk_embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7],
        [0.1, 0.1, 0.1],
        [0.9, 0.9, 0.9]
    ])
    # Dummy query embedding similar to chunk 3
    query_embedding = np.array([[0.5, 0.6, 0.7]])
    indices, scores = retrieve_top_k(query_embedding, chunk_embeddings, k=2)
    # We expect 2 indices to be returned
    assert len(indices) == 2