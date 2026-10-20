import json
import time

from langchain_core.documents import Document
from pinecone.openapi_support.exceptions import NotFoundException

from app import config
from app.chunking import chunk_documents
from app.enrichment import generate_image_caption
from app.parsers import load_pdfs
from app.retrieval import _get_vectorstore

CACHE_FILE = config.DATA_DIR / "enriched_cache.json"


def main():
    # 1. CHECK CACHE FIRST
    if CACHE_FILE.exists():
        print(
            f"Found cache at {CACHE_FILE}. Skipping OpenAI Vision calls to save time and money..."
        )
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        raw_docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in cached_data
        ]
    else:
        print("1. Loading PDFs & extracting text/images...")
        raw_docs = load_pdfs()

        print("\n2. Enriching Images with Vision LLM...")
        for i, doc in enumerate(raw_docs):
            if doc.metadata.get("type") == "image":
                print(
                    f"  -> Analyzing image {i}/{len(raw_docs)}: Page {doc.metadata['page']} of {doc.metadata['source']}"
                )
                caption = generate_image_caption(doc.metadata["image_path"])
                doc.page_content = f"Visual Data from Page {doc.metadata['page']} of {doc.metadata['source']}:\n{caption}"
                time.sleep(0.5)

        # Save to cache so we never pay for this again
        cached_data = [
            {"page_content": d.page_content, "metadata": d.metadata} for d in raw_docs
        ]
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cached_data, f)
        print(f"\nSaved enriched docs to cache: {CACHE_FILE}")

    print("\n3. Chunking text documents...")
    final_docs = chunk_documents(raw_docs)
    print(f"Total chunks ready for Pinecone: {len(final_docs)}")

    print("\n4. Pushing to Pinecone...")
    vectorstore = _get_vectorstore()

    # 3. SAFELY DELETE EXISTING VECTORS
    try:
        vectorstore.delete(delete_all=True)
        print("Cleared existing index.")
    except NotFoundException:
        print("Index is fresh/empty. Skipping deletion.")

    # 4. PUSH IN BATCHES
    batch_size = 100
    for i in range(0, len(final_docs), batch_size):
        batch = final_docs[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(
            f"  -> Uploaded batch {i // batch_size + 1} ({(i + len(batch))}/{len(final_docs)})"
        )

    print("\n✅ Ingestion Complete! All text and image captions are in Pinecone.")


if __name__ == "__main__":
    main()
