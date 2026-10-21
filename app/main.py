import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.graph import compiled_graph

app = FastAPI(
    title="Advanced Rerank RAG API",
    description="Reranker RAG system for Economic Survey of India.",
    version="0.1.0",
)


class QueryResquest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    final_response: str
    stage1_docs: list[dict]
    stage2_docs: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryResquest):
    try:
        result = compiled_graph.invoke({"query": request.query})

        # Stage 1.Hybrid retrieval for UI
        stage1 = [
            {
                "content": r.page_content[:300],
                "source": r.metadata.get("source", "unknown"),
                "type": r.metadata.get("type"),
            }
            for r in result.get("retrieved_docs", [])
        ]

        stage2 = [
            {
                "content": r.page_content[:300],
                "source": r.metadata.get("source", "unknown"),
                "type": r.metadata.get("type"),
                "score": r.metadata.get("rerank_score"),
            }
            for r in result.get("reranked_docs", [])
        ]

        return {
            "query": request.query,
            "final_response": result.get("final_response", ""),
            "stage1_docs": stage1,
            "stage2_docs": stage2,
        }

    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
