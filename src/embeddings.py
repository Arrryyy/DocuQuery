import os
import openai

# Retrieve your API key from the environment
openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    """
    Generate an embedding for a single text input using OpenAI's API.

    :param text: The text to embed.
    :param model: The embedding model to use.
    :return: A list of floats representing the embedding vector.
    """
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    # The embedding is returned in the first item of 'data'
    return response["data"][0]["embedding"]

def get_embeddings(texts: list[str], model: str = "text-embedding-ada-002") -> list[list[float]]:
    """
    Generate embeddings for a list of text inputs.

    :param texts: A list of texts to embed.
    :param model: The embedding model to use.
    :return: A list of embedding vectors.
    """
    response = openai.Embedding.create(
        input=texts,
        model=model
    )
    embeddings = [item["embedding"] for item in response["data"]]
    return embeddings

if __name__ == "__main__":
    # Example usage for a single text
    sample_text = "Hello, world!"
    embedding = get_embedding(sample_text)
    print("Embedding length:", len(embedding))
    print("First 5 elements of the embedding:", embedding[:5])
    
    # Example usage for multiple texts
    sample_texts = ["Hello, world!", "This is another sentence."]
    embeddings = get_embeddings(sample_texts)
    print("Generated", len(embeddings), "embeddings.")