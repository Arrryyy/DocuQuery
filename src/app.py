import os
from flask import Flask, request, render_template_string
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from pathlib import Path

# Initialize the Flask app
app = Flask(__name__)

# Hardcoded path to the chunks folder
CHUNKS_FOLDER = "./data/chunks"

# Load the SBERT model for embeddings
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_chunks(chunks_dir: str) -> list[str]:
    chunks_path = Path(chunks_dir)
    texts = []
    for file_path in sorted(chunks_path.glob("*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read().strip())
    return texts

def compute_embeddings(texts: list[str]) -> np.ndarray:
    return np.array(sbert_model.encode(texts))

def compute_query_embedding(query: str) -> np.ndarray:
    return np.array(sbert_model.encode([query]))

def retrieve_top_k(query_embedding: np.ndarray, chunk_embeddings: np.ndarray, k: int = 3):
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_k_idx = similarities.argsort()[-k:][::-1]
    return top_k_idx, similarities[top_k_idx]

def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Using the context below, answer the following question:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    return prompt

# Set up a free LLM using Hugging Face's distilgpt2 (runs on CPU here)
generator = pipeline("text-generation", model="distilgpt2", device=-1)

def generate_answer(prompt: str) -> str:
    # Use max_new_tokens to allow for generation beyond the prompt
    response = generator(prompt, max_new_tokens=50, num_return_sequences=1)
    return response[0]['generated_text']

# Load chunks and precompute embeddings at startup
chunks = load_chunks(CHUNKS_FOLDER)
if chunks:
    chunk_embeddings = compute_embeddings(chunks)
else:
    chunk_embeddings = np.array([])

print(f"Loaded {len(chunks)} chunks from {CHUNKS_FOLDER}.")

# Beautiful, Bootstrap-based HTML templates
HTML_FORM = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>RAG Query Interface</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body class="bg-light">
    <div class="container py-5">
      <h1 class="mb-4 text-center">Ask a Question</h1>
      {% if error %}
        <div class="alert alert-danger" role="alert">
          {{ error }}
        </div>
      {% endif %}
      <form method="post">
        <div class="mb-3">
          <label for="query" class="form-label">Enter your query:</label>
          <input type="text" class="form-control" id="query" name="query" placeholder="Type your question here...">
        </div>
        <button type="submit" class="btn btn-primary">Submit</button>
      </form>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
"""

HTML_RESULT = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>RAG Answer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body class="bg-light">
    <div class="container py-5">
      <h1 class="mb-4 text-center">Answer to: "{{ query }}"</h1>
      <div class="card shadow-sm">
        <div class="card-body">
          <p class="card-text">{{ answer }}</p>
        </div>
      </div>
      <div class="mt-4 text-center">
        <a href="/" class="btn btn-secondary">Ask Another Question</a>
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if not query:
            return render_template_string(HTML_FORM, error="Please enter a query.")
        
        if chunk_embeddings.size == 0:
            return render_template_string(HTML_FORM, error="No document chunks available. Please run the ingestion process.")
        
        query_embedding = compute_query_embedding(query)
        top_indices, scores = retrieve_top_k(query_embedding, chunk_embeddings, k=3)
        retrieved_chunks = [chunks[i] for i in top_indices]
        prompt = build_prompt(query, retrieved_chunks)
        answer = generate_answer(prompt)
        
        return render_template_string(HTML_RESULT, query=query, answer=answer)
    return render_template_string(HTML_FORM)

if __name__ == "__main__":
    app.run(debug=True)