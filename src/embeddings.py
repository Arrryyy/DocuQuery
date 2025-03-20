import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load a free SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single piece of text using Sentence Transformers.
    
    :param text: The text to embed.
    :return: A list of floats representing the embedding vector.
    """
    return model.encode(text).tolist()

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    
    :param texts: A list of text strings.
    :return: A list of embedding vectors.
    """
    return model.encode(texts).tolist()

def load_chunks(chunks_dir: str) -> list[str]:
    """
    Load all text chunks from the specified directory.
    
    :param chunks_dir: Directory where chunk text files are stored.
    :return: A list of text strings from the chunk files.
    """
    chunks_path = Path(chunks_dir)
    texts = []
    for file_path in sorted(chunks_path.glob("*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts

if __name__ == "__main__":
    # Set the path to your chunks folder (should be created by the ingestion script)
    chunks_folder = "./data/chunks"
    
    # Load the chunks from disk
    texts = load_chunks(chunks_folder)
    print(f"Loaded {len(texts)} chunks from {chunks_folder}.")
    
    # Generate embeddings for the chunks
    embeddings = get_embeddings(texts)
    print("Generated embeddings for the chunks.")
    if embeddings:
        print("First embedding length:", len(embeddings[0]))