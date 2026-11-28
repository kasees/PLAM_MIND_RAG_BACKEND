# Palm Mind AI — Document Ingestion & Conversational RAG Backend

Backend service built with **FastAPI**, **Qdrant**, **SQLite**, **Redis**, and **Groq LLM** for document ingestion and multi-turn conversational RAG with automated interview booking.

## Features

- **Document Ingestion API (`POST /ingest`)**: Accepts `.pdf` and `.txt` uploads, applies selectable chunking (`fixed` or `paragraph`), generates embeddings (`all-MiniLM-L6-v2`), stores vectors in Qdrant, and saves metadata in SQLite.
- **Conversational RAG API (`POST /chat`)**: Custom RAG pipeline (no `RetrievalQAChain`), Redis multi-turn chat memory, and LLM-based interview booking extraction (`name`, `email`, `date`, `time`) persisted in SQLite.

## Setup & Run (macOS)

1. **Start Redis**

   ```bash
   brew install redis
   brew services start redis
   ```

2. **Install Dependencies**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment**

   ```bash
   cp .env.example .env
   # Add your GROQ_API_KEY inside .env
   ```

4. **Start FastAPI Server**

   ```bash
   uvicorn main:app --reload
   ```

   Open **`http://127.0.0.1:8001`** to test `/ingest` and `/chat`.
