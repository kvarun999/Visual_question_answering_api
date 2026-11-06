import uuid
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.dependencies import get_vlm_service
from app.core.audit import audit_image_payload
from app.core.config import Settings, get_settings
from app.services.image import encode_image_to_base64, validate_image_integrity
from app.services.vlm import BaseVLMClient

router = APIRouter()


class VQAResponse(BaseModel):
    """VQA API standard response schema."""
    answer: str


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Readiness and health check endpoint."""
    return HealthResponse(status="healthy")


@router.post(
    "/api/vqa",
    response_model=VQAResponse,
    status_code=status.HTTP_200_OK,
    tags=["VQA"]
)
async def answer_visual_question(
    file: UploadFile = File(...),
    question: str = Form(...),
    vlm_client: BaseVLMClient = Depends(get_vlm_service),
    settings: Settings = Depends(get_settings)
) -> VQAResponse:
    """Validates the multimodal VQA request, audits image bytes, and queries the VLM.

    Args:
        file: Multipart uploaded image file.
        question: Text question string.
        vlm_client: Injected Vision-Language Model client.
        settings: Application settings.

    Returns:
        JSON response containing the model's grounded answer.

    Raises:
        HTTPException 400: If question is empty, file is missing, empty, invalid, or corrupted.
    """
    # 1. Validate question input: reject empty or whitespace-only text
    if not question or not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty or whitespace only."
        )

    # 2. Validate file existence and MIME type
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An image file is required."
        )

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file content type '{content_type}'. An image file is required."
        )

    # 3. Read image bytes and validate non-emptiness
    image_bytes = await file.read()
    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty."
        )

    # 4. Strict image integrity verification via PIL
    try:
        validate_image_integrity(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image: {exc}"
        )

    # 5. Cryptographic auditing: SHA-256 of raw received image bytes
    request_id = str(uuid.uuid4())
    audit_image_payload(request_id, image_bytes)

    # 6. Image preprocessing and Base64 encoding
    base64_image, normalized_mime = encode_image_to_base64(
        image_bytes=image_bytes,
        mime_type=content_type,
        max_dimension=settings.max_image_dimension
    )

    # 7. Query the Vision-Language Model
    answer = await vlm_client.ask_vision_model(
        base64_image=base64_image,
        mime_type=normalized_mime,
        question=question.strip()
    )

    return VQAResponse(answer=answer)

