import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "advanced-rerank-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"

if not OPENAI_API_KEY:
    raise RuntimeError("Please provide the OPENAI_API_KEY to proceed.")

if not PINECONE_API_KEY:
    raise RuntimeError("Please provide PINECONE_API_KEY to proceed.")
