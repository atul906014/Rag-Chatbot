# DocChat — Dual-Mode RAG Chatbot

DocChat is a web app for chatting with an LLM or asking questions about uploaded documents. Document answers use hybrid retrieval, cross-encoder reranking, and citations to the source file, page, and detected section.

## Features

- **General chat** for questions that do not need uploaded files.
- **Document chat** that retrieves relevant passages from uploaded files and answers using that context.
- Hybrid retrieval using FAISS vector search and BM25 keyword search.
- Cross-encoder reranking of retrieved passages.
- Source, page, and section citation display.
- PDF, DOCX, TXT, and Markdown (`.md`) uploads.
- Per-tab sessions, with idle-session cleanup.
- Optional PostgreSQL-backed conversation checkpoints. Document indexes remain in memory.

## Requirements

- Python 3.10 or later
- A [Groq API key](https://console.groq.com/keys)
- Windows, macOS, or Linux

The embedding and reranking models are downloaded on first use. The cross-encoder runs on CPU by default.

## Setup

Run these commands from the repository root in PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

Create a `.env` file in the repository root:

```env
GROQ_API_KEY=your_groq_api_key
```

`GROQ_API_KEY` is required. The remaining settings are optional:

| Variable | Default | Description |
| --- | --- | --- |
| `GROQ_MAIN_MODEL` | `llama-3.3-70b-versatile` | Groq chat model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Hugging Face sentence embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face cross-encoder model |
| `CHUNK_SIZE` | `800` | Document chunk size |
| `CHUNK_OVERLAP` | `120` | Overlap between document chunks |
| `RETRIEVE_K` | `20` | Number of hybrid retrieval candidates |
| `RERANK_TOP_K` | `4` | Number of reranked passages used to answer |
| `SESSION_IDLE_TTL_SECONDS` | `1800` | Idle session expiration time |
| `DATABASE_URL` | unset | Optional PostgreSQL URL for conversation checkpoints |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed CORS origins |

If PostgreSQL is not configured or unavailable, the app uses in-memory conversation checkpoints. Uploaded document indexes are in memory in either case.

## Run

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location backend
python -m uvicorn app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), and the health endpoint is [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).

## Run with Docker

Build and run the app from the repository root:

```powershell
docker build -t docchat .
docker run --rm -p 8000:8000 --env-file .env docchat
```

The `.env` file must define `GROQ_API_KEY`. The embedding and reranking models are downloaded on first startup and cached inside the container; deleting the container removes that cache.

## API overview

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Serve the web app |
| `/health` | GET | Health check |
| `/build_index` | POST | Upload and index one or more files (`session_id` and `files` form fields) |
| `/chat` | POST | General chat (`session_id` and `message` JSON fields) |
| `/docs_chat` | POST | Chat with the session's indexed documents |
| `/session/ping` | POST | Keep a session active |
| `/session/end` | POST | Clear session indexes and conversation state |

## Notes

- Scanned PDFs without a text layer are not OCR-processed and may not produce searchable text.
- PDF page numbers are displayed starting at 1.
- Section labels are inferred from recognizable headings; they may be absent or imperfect for documents without clear headings.
- Uploaded files are processed from temporary files that are removed after loading. Indexes are held in memory and cleared when a session ends or expires.

## License

This project is distributed under the MIT License. See [LICENSE](./LICENSE).