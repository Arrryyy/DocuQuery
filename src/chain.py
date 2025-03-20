import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Import functions from your embeddings script
from embeddings import load_chunks, get_embeddings

# Set up the free LLM text generation model from Hugging Face.
# We use 'distilgpt2' here because it's lightweight and free.
generator = pipeline("text-generation", model="distilgpt2")

def compute_query_embedding(query: str, model) -> np.ndarray:
    """
    Compute the embedding for the query using the provided SentenceTransformer model.
    """
    return np.array(model.encode([query]))

def retrieve_top_k(query_embedding: np.ndarray, chunk_embeddings: np.ndarray, k: int = 3):
    """
    Compute cosine similarity between the query embedding and chunk embeddings,
    then return indices of the top k most similar chunks along with their scores.
    """
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_k_idx = similarities.argsort()[-k:][::-1]
    return top_k_idx, similarities[top_k_idx]

def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    """
    Build a prompt by concatenating the retrieved context (chunks) and the user query.
    """
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Using the context below, answer the following question:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    return prompt

def generate_answer(prompt: str) -> str:
    """
    Generate an answer using the free LLM from Hugging Face.
    """
    # Use max_new_tokens to ensure additional tokens are generated beyond the input prompt.
    response = generator(prompt, max_new_tokens=150, num_return_sequences=1)
    return response[0]['generated_text']

def main():
    # Hardcoded path to the chunks folder.
    chunks_folder = "./data/chunks"
    
    # Load text chunks from the folder.
    texts = load_chunks(chunks_folder)
    if not texts:
        print("No chunk files found in", chunks_folder)
        return
    print(f"Loaded {len(texts)} chunks from {chunks_folder}.")
    
    # Generate embeddings for the chunks using your SentenceTransformer model.
    # (This function is from your embeddings module.)
    chunk_embeddings = np.array(get_embeddings(texts))
    
    # For the query embedding, instantiate the same model.
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    query = input("Enter your query: ")
    query_embedding = compute_query_embedding(query, sbert_model)
    
    # Retrieve top 3 relevant chunks.
    top_indices, scores = retrieve_top_k(query_embedding, chunk_embeddings, k=3)
    retrieved_chunks = [texts[i] for i in top_indices]
    
    print("\nRetrieved chunks:")
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        print(f"Chunk {idx} (Score: {scores[idx-1]:.4f}): {chunk[:100]}...")  # preview first 100 chars
    
    # Build the prompt with context and the query.
    prompt = build_prompt(query, retrieved_chunks)
    print("\nPrompt built for generation:")
    print(prompt)
    
    # Generate an answer using the prompt.
    answer = generate_answer(prompt)
    print("\nFinal Answer:")
    print(answer)

if __name__ == "__main__":
    main()