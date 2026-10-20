from app.parsers import load_pdfs


def main():
    # MUST have parentheses!
    docs = load_pdfs()

    text_docs = [d for d in docs if d.metadata["type"] == "text"]
    image_docs = [d for d in docs if d.metadata["type"] == "image"]

    print("\n====== Extraction Summary =======\n")
    print(f"Total text chunks/pages: {len(text_docs)}")
    print(f"Total image chunks/pages: {len(image_docs)}")

    if text_docs:
        print("\n===== SAMPLE TEXT =====")
        print(text_docs[0].page_content[:300] + "......")

    if image_docs:
        print("\n===== SAMPLE IMAGE INFO =====")
        print(f"Image saved at: {image_docs[0].metadata['image_path']}")


if __name__ == "__main__":
    main()
