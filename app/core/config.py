import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_submission_defaults() -> tuple[Optional[str], Optional[str]]:
    """Attempt to load default provider and model from submission.json if present."""
    submission_path = Path("submission.json")
    if submission_path.is_file():
        try:
            with open(submission_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("vlm_provider"), data.get("model_name")
        except Exception:
            pass
    return None, None


_default_provider, _default_model = _load_submission_defaults()


class Settings(BaseSettings):
    """Application settings and configuration parameters.
    
    All settings are configurable via environment variables or .env file.
    No secrets or provider names are hardcoded.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application Information
    app_name: str = "Visual Question Answering API"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # VLM Strategy Configuration
    vlm_provider: str = _default_provider or "openai"
    model_name: str = _default_model or "gpt-4o-mini"

    # OpenAI API Settings
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"

    # Anthropic API Settings
    anthropic_api_key: Optional[str] = None
    anthropic_base_url: str = "https://api.anthropic.com/v1"

    # Ollama / Local VLM Settings
    ollama_base_url: str = "http://localhost:11434"

    # Cryptographic Audit Settings
    audit_log_path: str = "audit.log"

    # Image Preprocessing & Network Settings
    max_image_dimension: int = 1024
    request_timeout: float = 60.0
    allowed_image_types: List[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif"
    ]

    # Prompt Engineering for Visual Grounding
    system_prompt: str = (
        "You are a visually grounded AI assistant. Answer the user's question accurately "
        "based ONLY on the provided image. Be concise. If the question asks for a count "
        "or number, reply with only the numerical digit."
    )


@lru_cache()
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()

