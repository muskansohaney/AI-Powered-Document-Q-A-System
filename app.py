import os
import io
import tempfile

from groq import Groq
from PyPDF2 import PdfReader
import numpy as np
import faiss

import gradio as gr

# --- Configuration ---
API_KEY = os.getenv("GROQ_API_KEY")
if API_KEY is None:
    raise ValueError("Set GROQ_API_KEY in environment")

client = Groq(api_key=API_KEY)

EMBEDDING_DIM = 768  # adjust if your embedding model has different size
TOP_K = 5  # number of chunks to retrieve for context

# --- Utility functions ---
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = []
    for page in reader.pages:
        p = page.extract_text()
        if p:
            text.append(p)
    return "\n".join(text)

def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> list[str]:
    # simple naïve tokenisation by words
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks

def embed_texts(texts: list[str]) -> np.ndarray:
    # You will need to plug in an embedding model of your choice.
    # For demo purposes only: random embeddings (NOT for production).
    # Replace with: e.g., sentence_transformers model.encode(texts)
    return np.random.randn(len(texts), EMBEDDING_DIM).astype("float32")

# --- Build Index (in-memory) ---
index = None
chunk_texts = []  # list of (orig_text, chunk_text)

def build_index_from_pdf_bytes(file_bytes: bytes):
    global index, chunk_texts
    full_text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(full_text)
    embeddings = embed_texts(chunks)
    # build FAISS index
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    chunk_texts = chunks

def retrieve_top_k(query: str, k: int = TOP_K) -> list[str]:
    q_emb = embed_texts([query])[0]
    D, I = index.search(np.array([q_emb]), k)
    results = [chunk_texts[i] for i in I[0]]
    return results

def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Based on the following context from a document:\n{context}\n\nAnswer the question:\n{question}"""

    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant for document Q&A."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant",  # adjust model as available
        temperature=0.0,
        max_tokens=512
    )
    return resp.choices[0].message.content

# --- Gradio UI ---
def process_pdf_and_qa(pdf_file, question):
    if pdf_file is None:
        return "Please upload a PDF first."

    # Gradio returns a NamedString or dict-like object with 'name' as file path
    file_path = pdf_file.name if hasattr(pdf_file, "name") else pdf_file["name"]

    # Open the file properly
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    build_index_from_pdf_bytes(file_bytes)
    top_chunks = retrieve_top_k(question, TOP_K)
    answer = generate_answer(question, top_chunks)
    return answer


with gr.Blocks() as demo:
    gr.Markdown("## PDF-based Q&A using Groq + Gradio")
    pdf_input = gr.File(label="Upload PDF", file_types=['.pdf'])
    question_input = gr.Textbox(label="Enter your question", placeholder="Ask anything about the document")
    answer_output = gr.Textbox(label="Answer", lines=10)
    btn = gr.Button("Ask")

    btn.click(fn=process_pdf_and_qa,
            inputs=[pdf_input, question_input],
            outputs=[answer_output])

demo.launch(share=True)

# --- Instructions to run ---
#pip install PyPDF2
#export GROQ_API_KEY="GROQ_API_KEY"
#python app.py