from fastapi import Depends
from app.core.config import Settings, get_settings
from app.services.vlm import BaseVLMClient, get_vlm_client


def get_vlm_service(settings: Settings = Depends(get_settings)) -> BaseVLMClient:
    """Dependency provider for the configured VLM client instance."""
    return get_vlm_client(settings)

