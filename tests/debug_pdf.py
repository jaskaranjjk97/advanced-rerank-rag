import fitz

from app import config


def main():
    # Get the first PDF it finds
    pdf_path = next(iter(config.PDF_DIR.glob("*.pdf")))
    print(f"Opening: {pdf_path.name}")

    doc = fitz.open(pdf_path)
    page = doc[0]  # Look at the first page only

    print("\n--- TEXT EXTRACTION TEST ---")
    text = page.get_text("text")
    print(f"Raw text length: {len(text)} characters")
    if len(text) > 0:
        print(f"Sample text: {text[:200]}...")
    else:
        print("WARNING: No text found. This PDF might be a scanned image.")

    print("\n--- IMAGE EXTRACTION TEST ---")
    # Note: This MUST be page.get_images(), not pdf_path.get_images()
    images = page.get_images(full=True)
    print(f"Found {len(images)} images on page 1.")


if __name__ == "__main__":
    main()
