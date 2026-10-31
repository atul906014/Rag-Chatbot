import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MAIN_MODEL = os.getenv("GROQ_MAIN_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", 20))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 4))
REJECT_THRESHOLD = float(os.getenv("REJECT_THRESHOLD", 0.0))

SESSION_IDLE_TTL = int(os.getenv("SESSION_IDLE_TTL_SECONDS", 1800))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")