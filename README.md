# AI-Powered Document Q&A System (Groq + Gradio + RAG)

An **AI-based document question-answering system** that allows users to upload a PDF and instantly ask questions about its content — powered by **Groq LLMs** and a custom **Retrieval-Augmented Generation (RAG)** pipeline.

---

## Features

- **PDF Upload & Parsing** – Extracts text from uploaded PDFs using `PyPDF2`.
- **Context Retrieval** – Performs document chunking and retrieves the most relevant sections using **FAISS**.
- **Intelligent Answering** – Uses **Groq API (Llama 3)** to generate accurate, context-grounded answers.
- **Lightweight RAG Implementation** – Built *without LangChain*, leveraging Python and the `updates` module for custom orchestration.
- **Gradio Frontend** – Simple and interactive UI for uploading PDFs and asking questions.
- **Modular Architecture** – Easily extendable with sentence embeddings or cloud vector DBs (like **Supabase Vector**).

---

## Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Frontend** | Gradio |
| **Backend** | Python, Groq API |
| **Vector Store** | FAISS |
| **Text Extraction** | PyPDF2 |
| **Architecture** | Retrieval-Augmented Generation (RAG) |

---

## System Workflow

1. **Upload a PDF** file through the Gradio interface.  
2. **Extract & chunk** the text into smaller pieces.  
3. **Embed & store** chunks into a FAISS vector index.  
4. **Query Processing:** When a user asks a question, relevant chunks are retrieved using vector similarity search.  
5. **Answer Generation:** Groq LLM generates a concise answer grounded in the document content.  
6. **Response Display:** The final answer is displayed on the Gradio UI.

---

## Example Use Cases

- Research Paper or Legal Document Summarization  
- Policy/Manual Understanding  
- Internal Knowledge Base Q&A  
- Academic or Business Document Analysis  

---

##  Installation

### 1️ Clone the Repository
```bash
git clone https://github.com/muskansohaney/AI-Powered-Document-Q-A-System.git
cd AI-Powered-Document-Q-A-System
