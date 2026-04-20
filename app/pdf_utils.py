
import base64
import fitz  # PyMuPDF


def pdf_to_images(pdf_bytes: bytes, dpi: int = 120) -> list[str]:
    """
    Convert each page of a PDF into a base64 PNG string.

    Returns a list where list[i] is the base64 image for page i (0-indexed).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[str] = []

    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
        png_bytes = pix.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        images.append(b64)

    doc.close()
    return images
