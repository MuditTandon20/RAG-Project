# RAG-Project

Production-grade RAG document assistant built with LangChain, FAISS, OpenAI, FastAPI, and Streamlit.

## Stack

- FastAPI backend
- Streamlit frontend
- LangChain
- OpenAI chat and embedding models
- FAISS vector store

## Setup

1. Create and activate a Python 3.10 environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=...
```

## Run

Start the backend:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Start the frontend:

```bash
python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

- Frontend: http://127.0.0.1:8501
- API docs: http://127.0.0.1:8000/docs

## Workflow

1. Upload a PDF or TXT file in the sidebar.
2. The backend loads, chunks, embeds, and stores the document in FAISS.
3. Ask a question in the chat box.
4. The app retrieves relevant chunks and asks the LLM to answer using only those chunks.

## Notes

- `.env`, uploaded files, logs, virtual environments, and generated vector indexes are intentionally not committed.
- OpenAI quota/billing must be available for indexing and querying.
