import time

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):  # light version model (~100 mb)
        print(f"Loading Cross Encoder Model: {model_name}")
        # This downloads the model once and caches it locally
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: Document, top_k: int = 5) -> list[Document]:
        """Takes a list of documents, scores them against the query, and returns the top K."""
        if not documents:
            return []

        start_time = time.time()

        # 1.creating pairs of query and documents
        pairs = [[query, doc.page_content] for doc in documents]

        # 2. Score all pairs (higher score = more relevant)
        scores = self.model.predict(pairs)

        print(f"Raw Scores: {[round(float(s), 2) for s in scores]}")

        # 3. Combine docs with scores
        doc_score_pairs = list(zip(documents, scores))

        # 4. Sort descending by score
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 5. Attach the score to metadata (so the UI can show it) and slice top_k
        reranked_docs = []
        for doc, score in doc_score_pairs[:top_k]:
            doc.metadata["rerank_score"] = float(score)
            reranked_docs.append(doc)

        elapsed = time.time() - start_time
        print(f"Reranked {len(documents)} docs in {elapsed:.3f} seconds.")

        return reranked_docs
