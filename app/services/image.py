import base64
import io
from typing import Optional, Tuple
from PIL import Image


def validate_image_integrity(image_bytes: bytes) -> None:
    """Validates that the provided bytes represent a valid, uncorrupted image.

    Raises:
        ValueError: If image bytes are empty or cannot be opened/verified as an image.
    """
    if not image_bytes:
        raise ValueError("Image data is empty.")

    try:
        # Image.open followed by verify checks file header and data integrity
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
    except Exception as exc:
        raise ValueError(f"Corrupted or invalid image data: {exc}") from exc


def encode_image_to_base64(
    image_bytes: bytes,
    mime_type: str,
    max_dimension: Optional[int] = 1024
) -> Tuple[str, str]:
    """Preprocesses the image (optional dimension capping) and converts it to a Base64 string.

    Args:
        image_bytes: Raw binary data of the image.
        mime_type: MIME type of the image (e.g. 'image/jpeg', 'image/png').
        max_dimension: Maximum allowed width or height in pixels. If image exceeds this,
                       it will be downscaled preserving aspect ratio.

    Returns:
        A tuple of (base64_string, normalized_mime_type).
    """
    # Open image for dimension inspection
    with Image.open(io.BytesIO(image_bytes)) as img:
        orig_format = img.format or "JPEG"
        width, height = img.size

        # If image dimensions are within limit, encode the raw bytes directly to save CPU
        if max_dimension is None or (width <= max_dimension and height <= max_dimension):
            b64_str = base64.b64encode(image_bytes).decode("utf-8")
            return b64_str, mime_type

        # Resize image preserving aspect ratio
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        # Convert palette or RGBA images if saving as JPEG
        out_format = orig_format.upper()
        if out_format in ("JPEG", "JPG"):
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            out_mime = "image/jpeg"
        elif out_format == "PNG":
            out_mime = "image/png"
        elif out_format == "WEBP":
            out_mime = "image/webp"
        else:
            # Fallback to JPEG
            if img.mode != "RGB":
                img = img.convert("RGB")
            out_format = "JPEG"
            out_mime = "image/jpeg"

        buffer = io.BytesIO()
        img.save(buffer, format=out_format, quality=90)
        resized_bytes = buffer.getvalue()
        b64_str = base64.b64encode(resized_bytes).decode("utf-8")
        return b64_str, out_mime

