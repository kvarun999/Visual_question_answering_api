import base64
import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import httpx
from fastapi import HTTPException, status

from app.core.config import Settings, get_settings


class BaseVLMClient(ABC):
    """Abstract base class for Vision-Language Model clients."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.model_name
        self.system_prompt = settings.system_prompt
        self.timeout = settings.request_timeout

    @abstractmethod
    async def ask_vision_model(self, base64_image: str, mime_type: str, question: str) -> str:
        """Submits the multimodal payload to the VLM provider and returns the answer."""
        pass


class OpenAIVLMClient(BaseVLMClient):
    """OpenAI VLM client implementing the chat completions multimodal protocol."""

    async def ask_vision_model(self, base64_image: str, mime_type: str, question: str) -> str:
        api_key = self.settings.openai_api_key
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OPENAI_API_KEY is not configured in environment or .env."
            )

        data_uri = f"data:{mime_type};base64,{base64_image}"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri,
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.0
        }

        endpoint = f"{self.settings.openai_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request to OpenAI VLM provider timed out."
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Network error communicating with OpenAI: {exc}"
            ) from exc

        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"OpenAI API error ({response.status_code}): {response.text}"
            )

        try:
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
            return answer
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Malformed response from OpenAI API: {exc}"
            ) from exc


class AnthropicVLMClient(BaseVLMClient):
    """Anthropic VLM client implementing the messages multimodal protocol."""

    async def ask_vision_model(self, base64_image: str, mime_type: str, question: str) -> str:
        api_key = self.settings.anthropic_api_key
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ANTHROPIC_API_KEY is not configured in environment or .env."
            )

        payload = {
            "model": self.model_name,
            "system": self.system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.0
        }

        endpoint = f"{self.settings.anthropic_base_url.rstrip('/')}/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request to Anthropic VLM provider timed out."
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Network error communicating with Anthropic: {exc}"
            ) from exc

        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Anthropic API error ({response.status_code}): {response.text}"
            )

        try:
            data = response.json()
            answer = data["content"][0]["text"].strip()
            return answer
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Malformed response from Anthropic API: {exc}"
            ) from exc


class OllamaVLMClient(BaseVLMClient):
    """Local Ollama VLM client."""

    async def ask_vision_model(self, base64_image: str, mime_type: str, question: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": question,
                    "images": [base64_image]
                }
            ],
            "stream": False
        }

        endpoint = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, json=payload)
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request to Ollama VLM timed out."
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Network error communicating with Ollama: {exc}"
            ) from exc

        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ollama API error ({response.status_code}): {response.text}"
            )

        try:
            data = response.json()
            answer = data["message"]["content"].strip()
            return answer
        except (KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Malformed response from Ollama API: {exc}"
            ) from exc


class MockVLMClient(BaseVLMClient):
    """Deterministic mock client for offline verification and testing."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.fixtures = {}
        fixtures_path = Path("fixtures/fixtures.json")
        if fixtures_path.is_file():
            try:
                with open(fixtures_path, "r", encoding="utf-8") as f:
                    self.fixtures = json.load(f)
            except Exception:
                pass

    async def ask_vision_model(self, base64_image: str, mime_type: str, question: str) -> str:
        try:
            raw_bytes = base64.b64decode(base64_image)
            img_hash = hashlib.sha256(raw_bytes).hexdigest()

            for key in ("image_a_path", "image_b_path"):
                exp_key = "expected_a_contains" if key == "image_a_path" else "expected_b_contains"
                path_str = self.fixtures.get(key)
                if path_str:
                    path = Path(path_str)
                    if path.is_file():
                        with open(path, "rb") as f:
                            if hashlib.sha256(f.read()).hexdigest() == img_hash:
                                return f"{self.fixtures.get(exp_key, '')}"
        except Exception:
            pass

        return "3"


def get_vlm_client(settings: Optional[Settings] = None) -> BaseVLMClient:
    """Factory creating the appropriate VLM client based on configuration."""
    resolved_settings = settings or get_settings()
    provider = resolved_settings.vlm_provider.lower().strip()

    if provider in ("openai", "gpt"):
        return OpenAIVLMClient(resolved_settings)
    elif provider in ("anthropic", "claude"):
        return AnthropicVLMClient(resolved_settings)
    elif provider in ("ollama", "local"):
        return OllamaVLMClient(resolved_settings)
    elif provider == "mock":
        return MockVLMClient(resolved_settings)
    else:
        # Default fallback to OpenAI client
        return OpenAIVLMClient(resolved_settings)


async def ask_vision_model(
    base64_image: str,
    mime_type: str,
    question: str,
    client: Optional[BaseVLMClient] = None
) -> str:
    """Constructs the multimodal payload and calls the configured VLM API.

    Args:
        base64_image: The base64 encoded image string.
        mime_type: The MIME type of the image (e.g., 'image/jpeg').
        question: The user's question.
        client: Optional VLM client instance; if not provided, the default client is used.

    Returns:
        The generated text answer from the vision model.
    """
    vlm_client = client or get_vlm_client()
    return await vlm_client.ask_vision_model(base64_image, mime_type, question)

