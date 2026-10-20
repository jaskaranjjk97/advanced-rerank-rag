from app.chunking import chunk_documents
from app.enrichment import generate_image_caption
from app.parsers import load_pdfs


def main():
    print("1. Loading raw PDF's")
    docs = load_pdfs()

    # Grab just the first image document to test
    image_doc = next(
        d for d in docs if d.metadata["type"] == "image" and d.metadata["page"] >= 5
    )
    print(
        f"\n2. Generating caption for: {image_doc.metadata['source']} Page {image_doc.metadata['page']}"
    )
    caption = generate_image_caption(image_doc.metadata["image_path"])

    print("\n=== CAPTION GENERATED ===")
    print(caption)
    print("=========================")

    # 3. Update the image document with the caption
    image_doc.page_content = (
        f"Visual Data from Page {image_doc.metadata['page']}: {caption}"
    )

    print("\n4. Testing Chunking...")
    final_docs = chunk_documents(docs)

    text_count = sum(1 for d in final_docs if d.metadata["type"] == "text")
    image_count = sum(1 for d in final_docs if d.metadata["type"] == "image")

    print(f"After chunking: {text_count} text chunks, {image_count} image chunks.")
    print("Ready for Ingestion!")


if __name__ == "__main__":
    main()
