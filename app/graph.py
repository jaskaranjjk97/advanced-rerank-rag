import json

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app import config
from app.chunking import chunk_documents
from app.llm import get_llm
from app.reranker import Reranker
from app.retrieval import build_bm25_index, hybrid_retrieval
from app.state import RAGState


def initialize_bm25():
    """Load cache docs and build bm25 index for stage 1."""
    CACHE_FILE = config.DATA_DIR / "enriched_cache.json"
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as file:
            cached_data = json.load(file)

        raw_docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in cached_data
        ]

        chunks = chunk_documents(raw_docs)
        build_bm25_index(chunks)


def retrieve_node(state: RAGState) -> dict:
    """LangGraph Node: retrieves the relevant docs based on hybrid retrieval."""
    query = state["query"]
    docs = hybrid_retrieval(query=query, k=20)
    print(f"  -> Retrieved {len(docs)} docs from Hybrid Search.")

    return {
        "retrieved_docs": docs,
        "trace": state.get("trace", [])
        + [
            {
                "node": "retrieve",
                "status": "ok",
                "details": {"count": len(docs)},
            }
        ],
    }


def rerank_node(state: RAGState) -> dict:
    """LangGraph Node: rerank the retrieved docs based on cross-encoder reranker."""
    query = state.get("query")
    docs = state.get("retrieved_docs", [])
    print(f"  -> Rerank Node received {len(docs)} docs.")

    if not docs:
        return {
            "reranked_docs": [],
            "trace": state.get("trace", [])
            + [
                {
                    "node": "rerank",
                    "status": "error",
                    "details": {"error": "No docs received"},
                }
            ],
        }

    reranker = Reranker()
    reranked_docs = reranker.rerank(query, documents=docs, top_k=5)

    top_score = reranked_docs[0].metadata.get("rerank_score", 0) if reranked_docs else 0

    print(f"  -> Reranked to top {len(reranked_docs)} docs. Top score: {top_score:.2f}")

    return {
        "reranked_docs": reranked_docs,
        "trace": state.get("trace", [])
        + [{"node": "rerank", "status": "ok", "details": {"top_score": top_score}}],
    }


def generation_node(state: RAGState) -> dict:
    """LangGraph Node: generates the response based on reranked docs."""

    query = state["query"]
    docs = state.get("reranked_docs", [])

    if not docs:
        return {
            "final_response": "I'm sorry, I could not find any relevant documents to answer your question.",
            "trace": state.get("trace", [])
            + [
                {
                    "node": "generate",
                    "status": "error",
                    "details": {"error": "No docs to generate from"},
                }
            ],
        }

    # Format context, including metadata for citations
    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        doc_type = doc.metadata.get("type", "text")
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_blocks.append(
            f"[{i}] Source:{source} (Page:{page}, Type:{doc_type})\n {doc.page_content}"
        )

    context_text = "\n\n---\n\n".join(context_blocks)

    system_prompt = """You are an expert economic analyst. Use the provided context to answer the user's question. 
    If the context contains image captions or table data, use that information to provide precise numbers and trends.
    Always cite your sources using the bracketed numbers [1], [2], etc."""

    user_prompt = f"Context: {context_text}\n\n Question: {query}"

    llm = get_llm(temperature=0.2)
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    return {
        "final_response": response.content,
        "trace": state.get("trace", [])
        + [
            {
                "node": "genration",
                "status": "ok",
                "details": {"length": len(response.content)},
            }
        ],
    }


def build_graph():
    initialize_bm25()
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generation", generation_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generation")
    graph.add_edge("generation", END)

    return graph.compile()


compiled_graph = build_graph()
