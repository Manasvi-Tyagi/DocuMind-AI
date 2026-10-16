"""Page rendering and local text extraction for multimodal analysis."""

import base64
import io
from dataclasses import dataclass, field

import pymupdf
import pytesseract
from PIL import Image, ImageOps

from .config import get_settings


@dataclass
class DocumentPage:
    number: int
    image: bytes
    text: str
    method: str
    warning: str = ""
    word_boxes: list = field(default_factory=list)

    @property
    def data_url(self):
        return "data:image/jpeg;base64," + base64.b64encode(self.image).decode("ascii")


def page_count(file):
    if file.name.lower().endswith(".pdf"):
        with pymupdf.open(stream=file.getvalue(), filetype="pdf") as pdf:
            return len(pdf)
    return 1


def _prepare_page(number, image, text=""):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((1800, 1800))
    method = "PDF text" if text.strip() else "Tesseract OCR"
    warning = ""
    if not text.strip():
        executable = get_settings().tesseract_cmd
        if executable:
            pytesseract.pytesseract.tesseract_cmd = executable
        try:
            text = pytesseract.image_to_string(
                ImageOps.autocontrast(ImageOps.grayscale(image)), lang="eng"
            ).strip()
        except (
            pytesseract.TesseractNotFoundError,
            pytesseract.TesseractError,
        ) as error:
            warning = f"OCR unavailable on page {number}: {type(error).__name__}. Vision can still read the image."
            method = "Image only"
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return DocumentPage(number, output.getvalue(), text.strip(), method, warning)


def prepare_document(file, page_numbers, max_pages=3):
    """Prepare only explicitly selected pages; never silently omit PDF pages."""
    numbers = list(page_numbers)
    if not numbers or len(numbers) > max_pages or len(set(numbers)) != len(numbers):
        raise ValueError(f"Select between one and {max_pages} distinct pages.")
    if file.name.lower().endswith(".pdf"):
        with pymupdf.open(stream=file.getvalue(), filetype="pdf") as pdf:
            if any(n < 1 or n > len(pdf) for n in numbers):
                raise ValueError("Selected page is outside the document.")
            pages = []
            for number in numbers:
                page = pdf[number - 1]
                scale = min(2.0, 1800 / max(page.rect.width, page.rect.height))
                pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                prepared = _prepare_page(number, image, page.get_text())
                for word in page.get_text("words", sort=True):
                    rect = pymupdf.Rect(word[:4]) * page.rotation_matrix
                    prepared.word_boxes.append(
                        {
                            "text": word[4],
                            "box": [
                                rect.x0 / page.rect.width,
                                rect.y0 / page.rect.height,
                                rect.x1 / page.rect.width,
                                rect.y1 / page.rect.height,
                            ],
                        }
                    )
                pages.append(prepared)
            return pages
    if numbers != [1]:
        raise ValueError("Images have only one page.")
    with Image.open(io.BytesIO(file.getvalue())) as image:
        return [_prepare_page(1, image)]


def pages_text(pages):
    return "\n\n".join(f"Page {page.number}:\n{page.text}" for page in pages)
