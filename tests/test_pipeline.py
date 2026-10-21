import json

from langchain_core.documents import Document

from app import config
from app.chunking import chunk_documents
from app.reranker import Reranker
from app.retrieval import build_bm25_index, hybrid_retrieval

CACHE_FILE = config.DATA_DIR / "enriched_cache.json"


def main():
    # 1. Load from Cache and Chunk
    print("Loading cache and building BM25 index...")
    with open(CACHE_FILE, encoding="utf-8") as file:
        cached_data = json.load(file)

    raw_docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in cached_data
    ]

    chunks = chunk_documents(raw_docs)
    build_bm25_index(chunks)

    # 2. Define a complex query
    query = "What is the trend in direct tax collections for FY26?"
    print(f"\nQuery: '{query}'")

    # 3. STAGE 1: Hybrid Retrieval (Fetch top 10)
    print("\n--- STAGE 1: Hybrid Retrieval (Top 5 shown) ---")
    stage1_docs = hybrid_retrieval(query, k=10)

    for i, doc in enumerate(stage1_docs[:5], 1):
        doc_type = "IMAGE CAPTION" if doc.metadata.get("type") == "image" else "TEXT"
        print(f"{i}. [{doc_type}] {doc.page_content[:150]}...")

    # 4. STAGE 2: Cross-Encoder Reranking
    print("\n--- STAGE 2: Cross-Encoder Reranked (Top 3) ---")
    reranker = Reranker()
    stage2_docs = reranker.rerank(query, stage1_docs, top_k=3)

    for i, doc in enumerate(stage2_docs, 1):
        doc_type = "IMAGE CAPTION" if doc.metadata.get("type") == "image" else "TEXT"
        score = doc.metadata.get("rerank_score", 0.0)
        print(f"{i}. (Score: {score:.2f}) [{doc_type}] {doc.page_content[:150]}...")


if __name__ == "__main__":
    main()
