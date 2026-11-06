# Grounded Visual Question Answering (VQA) API

A production-grade, grounded Visual Question Answering microservice built with **FastAPI** and **Vision-Language Models (VLMs)**.

This service eliminates the "Caption-then-Guess" hallucination problem by enforcing visual grounding: answers strictly anchor to the uploaded pixels and dynamically change when the image changes, even when the question is identical.

---

## Key Features

- **Strict Visual Grounding**: Direct multimodal payload generation without intermediate generic captioning or ungrounded language priors.
- **Cryptographic Auditing**: Calculates an SHA-256 hash of raw incoming image bytes before any preprocessing and appends it to `audit.log` along with a unique request ID, proving the image data was received and not dropped.
- **Strict Input Validation**: Rejects empty questions, missing files, non-image MIME types, zero-byte uploads, and corrupted image payloads with explicit `4xx` responses.
- **Multi-Provider Architecture**: Supports OpenAI (`gpt-4o`, `gpt-4o-mini`), Anthropic (`claude-3-5-sonnet-20241022`), Ollama (`llava`), and deterministic mock modes for offline testing.
- **Zero Hardcoding**: Every parameter (provider, model, base URLs, timeouts, dimensions, system prompt, audit path) is dynamically loaded via Pydantic `BaseSettings` and environment variables.
- **Docker Ready**: Complete `Dockerfile` and `docker-compose.yml` with automated container health checking.

---

## Architecture Overview

```
Client Application
       │
       ▼ [POST /api/vqa (multipart: file + question)]
FastAPI Router (app/api/router.py)
       │
       ├──► Input Validation Layer (empty string check, image integrity verification)
       │
       ├──► Cryptographic Auditor (SHA-256 computation ➔ append to audit.log)
       │
       ├──► Image Preprocessor & Base64 Encoder (app/services/image.py)
       │
       └──► VLM API Client (app/services/vlm.py)
               │
               ▼ [Multimodal JSON Payload]
       Vision-Language Model Provider (OpenAI / Anthropic / Local)
               │
               ▼
       JSON Response: { "answer": "..." }
```

---

## Directory Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application initialization & middleware
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py        # /api/vqa & /health endpoint definitions
│   │   └── dependencies.py  # Dependency injection for VLM client
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Pydantic BaseSettings environment configuration
│   │   └── audit.py         # SHA-256 cryptographic audit logger
│   └── services/
│       ├── __init__.py
│       ├── image.py         # Image validation, resizing & Base64 encoding
│       └── vlm.py           # Multi-provider multimodal VLM integration
├── fixtures/
│   ├── fixtures.json        # Visual grounding test definition
│   ├── image_a.jpg          # Fixture A (3 red apples on table)
│   └── image_b.jpg          # Fixture B (5 red apples on table)
├── tests/
│   ├── __init__.py
│   └── test_api.py          # Comprehensive test suite covering all requirements
├── .env.example             # Documented environment variables template
├── .gitignore               # Git ignore rules
├── docker-compose.yml       # Production container orchestration
├── Dockerfile               # Container build configuration
├── requirements.txt         # Pinned Python dependencies
├── submission.json          # Automated evaluator model strategy configuration
└── README.md
```

---

## Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `VLM_PROVIDER` | Provider strategy (`openai`, `anthropic`, `ollama`, `mock`) | `openai` |
| `MODEL_NAME` | Model identifier | `gpt-4o-mini` |
| `OPENAI_API_KEY` | OpenAI API key | *(empty / placeholder)* |
| `OPENAI_BASE_URL` | OpenAI base endpoint URL | `https://api.openai.com/v1` |
| `ANTHROPIC_API_KEY`| Anthropic API key | *(empty / placeholder)* |
| `ANTHROPIC_BASE_URL`| Anthropic base endpoint URL | `https://api.anthropic.com/v1` |
| `OLLAMA_BASE_URL` | Ollama local server base URL | `http://localhost:11434` |
| `AUDIT_LOG_PATH` | File path for cryptographic audit logging | `audit.log` |
| `MAX_IMAGE_DIMENSION` | Max pixel dimension before downscaling | `1024` |
| `REQUEST_TIMEOUT` | VLM API request timeout in seconds | `60.0` |
| `SYSTEM_PROMPT` | Grounding system prompt | Concise grounded instruction |

---

## API Endpoints

### 1. Health Check
- **Method**: `GET /health`
- **Response**: `200 OK`
```json
{
  "status": "healthy"
}
```

### 2. Visual Question Answering
- **Method**: `POST /api/vqa`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Image file (`image/jpeg`, `image/png`, `image/webp`)
  - `question`: Plain text question string
- **Response**: `200 OK`
```json
{
  "answer": "3"
}
```

### Error Responses
- `400 Bad Request`: If question is empty/whitespace, file is omitted/corrupt/non-image.
- `422 Unprocessable Entity`: If required form fields are missing from multipart request.

---

## Visual Grounding Fixtures

The repository includes controlled visual grounding fixtures in `fixtures/`:

- `fixtures/image_a.jpg`: Exactly 3 red apples on a wooden table.
- `fixtures/image_b.jpg`: Exactly 5 red apples on the same wooden table.
- `fixtures/fixtures.json`:
```json
{
  "question": "How many red apples are on the table? Respond with only the number.",
  "image_a_path": "fixtures/image_a.jpg",
  "expected_a_contains": "3",
  "image_b_path": "fixtures/image_b.jpg",
  "expected_b_contains": "5"
}
```

When evaluated, the exact same question is asked against both images:
- For `image_a.jpg` ➔ model output contains `"3"`.
- For `image_b.jpg` ➔ model output contains `"5"`.
- Output strictly diverges based on visual input.

---

## Cryptographic Audit Logging

Every incoming image accepted by the endpoint is intercepted before preprocessing:
1. `hashlib.sha256(image_bytes).hexdigest()` computes the cryptographic hash.
2. An audit record is appended to `audit.log`:
   ```
   <ISO_8601_TIMESTAMP> | <REQUEST_UUID> | <SHA-256_HEX_DIGEST>
   ```
This provides non-repudiation proof that the exact pixel bytes were received in memory.

---

## Running with Docker Compose

To start the service with containerized health checks:

```bash
docker compose up -d --build
```

Verify service status:
```bash
docker compose ps
curl http://localhost:8000/health
```

---

## Running Locally

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. Run tests:
   ```bash
   pytest -v
   ```
