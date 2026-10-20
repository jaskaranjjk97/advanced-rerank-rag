import fitz  # PyMuPDF
from langchain_core.documents import Document

from app import config


def load_pdfs() -> list[Document]:
    """Extract text and page images from all PDFs."""
    docs = []

    img_dir = config.DATA_DIR / "extracted_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in config.PDF_DIR.glob("*.pdf"):
        print(f"\nProcessing: {pdf_path.name}")
        pdf = fitz.open(pdf_path)

        for page_num, page in enumerate(pdf, start=1):
            # 1. EXTRACT TEXT (Using blocks for better formatting)
            blocks = page.get_text("blocks")
            # block[4] is the text string, block[6] == 0 means it's a text block
            text = " ".join([b[4] for b in blocks if b[6] == 0])
            text = " ".join(text.split())  # Clean whitespace

            if len(text) > 50:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num,
                            "type": "text",
                        },
                    )
                )
                print(f"  -> Extracted text from page {page_num} ({len(text)} chars)")

            # 2. RENDER PAGE AS IMAGE (Captures all charts, tables, and graphics)
            # 150 DPI is a good balance between readability and file size
            pix = page.get_pixmap(dpi=150)
            image_filename = f"{pdf_path.stem}_p{page_num}.png"
            image_path = img_dir / image_filename
            pix.save(image_path)

            docs.append(
                Document(
                    page_content=f"[Visual page {page_num} from {pdf_path.name}]",
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num,
                        "type": "image",
                        "image_path": str(image_path),
                    },
                )
            )
            print(f"  -> Saved page image: {image_filename}")

    return docs
