import io
import pymupdf
import pytesseract
from PIL import Image
from .config import get_settings


def extract_document_text(file):
    """Extract selectable PDF text or use English OCR."""
    tesseract_path = get_settings().tesseract_cmd
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    file_bytes = file.getvalue()
    extracted_pages = []
    if file.name.lower().endswith(".pdf"):
        document = pymupdf.open(stream=file_bytes, filetype="pdf")
        try:
            for page_number, page in enumerate(document):
                page_text = page.get_text().strip()
                if not page_text:
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                    page_text = pytesseract.image_to_string(image, lang="eng").strip()
                if page_text:
                    extracted_pages.append(f"Page {page_number + 1}:\n{page_text}")
        finally:
            document.close()
    else:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        page_text = pytesseract.image_to_string(image, lang="eng").strip()
        if page_text:
            extracted_pages.append(f"Image content:\n{page_text}")
    return "\n\n".join(extracted_pages)
