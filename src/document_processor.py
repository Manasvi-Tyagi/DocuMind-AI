"""Step 2 — Document preprocessing.

* PDF pages are rendered to images (PyMuPDF).
* Every page image is normalised to a common resolution.
* Blank margins can optionally be cropped (OpenCV).
"""

from dataclasses import dataclass
from io import BytesIO

import cv2
import pymupdf
import numpy as np
from PIL import Image, ImageOps, ImageSequence

IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp")


@dataclass
class DocumentPage:
    page_number: int
    image: Image.Image
    # Text layer embedded in the PDF ("" for images / scanned PDFs).
    embedded_text: str = ""


def normalize_resolution(image: Image.Image, target_long_side: int) -> Image.Image:
    """Resize so the longest side equals ``target_long_side`` pixels."""

    width, height = image.size
    long_side = max(width, height)
    if long_side == target_long_side or long_side == 0:
        return image

    scale = target_long_side / long_side
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resample = Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.BICUBIC
    return image.resize(new_size, resample)


def crop_margins(
    image: Image.Image,
    padding: int = 20,
    min_content_fraction: float = 0.2,
) -> Image.Image:
    """Remove blank borders around the content.

    The crop is skipped ("where appropriate") when the detected content box
    is tiny — e.g. an almost empty page — because cropping would then throw
    away the context of the page.
    """

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    # Dark content on light background -> content becomes white (non-zero).
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Remove isolated specks (scanner noise) so they don't stretch the box.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    points = cv2.findNonZero(mask)
    if points is None:
        return image

    x, y, w, h = cv2.boundingRect(points)
    img_w, img_h = image.size
    if (w * h) < min_content_fraction * img_w * img_h:
        return image

    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(img_w, x + w + padding)
    bottom = min(img_h, y + h + padding)
    return image.crop((left, top, right, bottom))


def preprocess_image(
    image: Image.Image,
    target_long_side: int,
    remove_margins: bool,
) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if remove_margins:
        image = crop_margins(image)
    return normalize_resolution(image, target_long_side)


def process_pdf(
    pdf_bytes: bytes,
    max_pages: int,
    dpi: int,
    target_long_side: int,
    remove_margins: bool,
) -> list[DocumentPage]:
    """Render each PDF page to an image and keep its embedded text."""

    pages = []
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
            if pdf.page_count == 0:
                raise ValueError("The PDF contains no pages.")
            if pdf.page_count > max_pages:
                raise ValueError(
                    f"The PDF has {pdf.page_count} pages. The current limit is {max_pages}."
                )

            zoom = dpi / 72  # PDF user space is 72 points per inch
            for index, page in enumerate(pdf):
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                image = Image.open(BytesIO(pixmap.tobytes("png")))
                pages.append(
                    DocumentPage(
                        page_number=index + 1,
                        image=preprocess_image(image, target_long_side, remove_margins),
                        embedded_text=page.get_text("text").strip(),
                    )
                )
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ValueError("Could not read this PDF. Check that it is a valid PDF file.") from exc

    return pages


def process_image(
    image_bytes: bytes,
    target_long_side: int,
    remove_margins: bool,
) -> list[DocumentPage]:
    """Load an image file. Multi-frame TIFFs become multiple pages."""

    try:
        source = Image.open(BytesIO(image_bytes))
    except Exception as exc:  # PIL raises several exception types
        raise ValueError("Could not read this image file.") from exc

    pages = []
    for index, frame in enumerate(ImageSequence.Iterator(source)):
        pages.append(
            DocumentPage(
                page_number=index + 1,
                image=preprocess_image(frame.copy(), target_long_side, remove_margins),
            )
        )
    return pages


def process_document(
    filename: str,
    data: bytes,
    max_pages: int,
    dpi: int,
    target_long_side: int,
    remove_margins: bool,
) -> list[DocumentPage]:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension == "pdf":
        return process_pdf(data, max_pages, dpi, target_long_side, remove_margins)
    if extension in IMAGE_EXTENSIONS:
        return process_image(data, target_long_side, remove_margins)[:max_pages]
    raise ValueError(f"Unsupported file type: .{extension}")
