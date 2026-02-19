import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

# Paths
CHROMA_DIR       = BASE_DIR / "chroma_db"
USER_MEMORY_FILE = BASE_DIR / "USER_MEMORY.md"
COMPANY_MEMORY_FILE = BASE_DIR / "COMPANY_MEMORY.md"
SAMPLE_DOCS_DIR  = BASE_DIR / "sample_docs"
ARTIFACTS_DIR    = BASE_DIR / "artifacts"

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# RAG settings
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80
TOP_K         = 5

# Memory
MEMORY_CONFIDENCE_THRESHOLD = 0.65

# Sandbox
SANDBOX_TIMEOUT = 30  # seconds — raised to accommodate rich analytics script
