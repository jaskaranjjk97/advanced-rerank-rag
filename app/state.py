from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    query: str

    # stage1->Hybrid retrieve
    retrieved_docs: list[Document]

    # Stage 2: Reranking
    reranked_docs: list[Document]

    # Generation
    final_response: str

    # Observability
    trace: list[dict]
