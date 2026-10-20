import hashlib

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from app import config

# --- Global variables for BM25 (kept in memory) ---
_bm25_index = None
_bm25_docs = []


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _build_bm25_index(docs: list[Document]):
    """create a bm25 index for keyword serach."""
    global _bm25_index, _bm25_docs
    _bm25_docs = docs
    _bm25_index = BM25Okapi([_tokenize(d.page_content) for d in docs])
    print(f"BM25 index created for {len(docs)} chunks.")


# Pinecone index
def _get_vectorstore() -> PineconeVectorStore:
    """Connect to the Pinecone index."""
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)
    embeddings = OpenAIEmbeddings(
        api_key=config.OPENAI_API_KEY,
        model=config.EMBEDDING_MODEL,
    )
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )


# --- Stage 1: Hybrid Retrieval ---


def hybrid_retrieval(query: str, k: int = 20) -> list[Document]:
    """Retrieve top-k from Pinecone and BM25, fused with Reciprocal Rank Fusion (RRF)."""
    # 1.Dense Retrieval
    vectorstore = _get_vectorstore()
    dense_docs = vectorstore.similarity_search(query, k=k)

    # 2.sparce docs
    if _bm25_index is None:
        raise ValueError("BM25 index not built. Call build_bm25_index() first.")
    tokenize_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenize_query)
    top_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:k]
    sparce_docs = [_bm25_docs[i] for i in top_indices]

    # 3.Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    doc_map = {}

    def _key(doc: Document) -> str:
        # Use hash of full content to prevent header collisions
        return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

    for ranked_list in (dense_docs, sparce_docs):
        for rank, doc in enumerate(ranked_list, start=1):
            key = _key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (60 + rank)
            doc_map[key] = doc

    # Sort by RRF score and return top k
    fused = sorted(rrf_scores.item(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in fused][:k]
