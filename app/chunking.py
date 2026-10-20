from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Chunk text documents, but keep image documents intact."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    final_docs = []

    for doc in docs:
        if doc.metadata["type"] == "text":
            # split the text into smaller chunks
            chunks = splitter.split_documents([doc])
            final_docs.extend(chunks)
        else:
            # Keep image documents (and their future captions) as single chunks
            final_docs.append(doc)

    return final_docs
