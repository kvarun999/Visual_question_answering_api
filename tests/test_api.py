import hashlib
import json
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_vlm_service
from app.core.config import Settings, get_settings
from app.main import app
from app.services.vlm import MockVLMClient


@pytest.fixture(autouse=True)
def override_vlm_service():
    """Ensure tests run against deterministic mock VLM client by default."""
    settings = get_settings()
    mock_client = MockVLMClient(settings)
    app.dependency_overrides[get_vlm_service] = lambda: mock_client
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_req1_fixtures_structure():
    """Requirement 1: Verify test fixtures directory and JSON definition."""
    fixtures_json_path = Path("fixtures/fixtures.json")
    assert fixtures_json_path.is_file(), "fixtures/fixtures.json must exist"

    with open(fixtures_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for field in ("question", "image_a_path", "expected_a_contains", "image_b_path", "expected_b_contains"):
        assert field in data, f"Missing required field: {field}"
        assert isinstance(data[field], str), f"Field {field} must be a string"
        assert len(data[field].strip()) > 0, f"Field {field} must not be empty"

    image_a = Path(data["image_a_path"])
    image_b = Path(data["image_b_path"])
    assert image_a.is_file(), f"image_a not found at {image_a}"
    assert image_b.is_file(), f"image_b not found at {image_b}"
    assert data["expected_a_contains"] != data["expected_b_contains"], (
        "expected_a_contains and expected_b_contains must be distinct"
    )


def test_req2_validation_empty_or_whitespace_question(client):
    """Requirement 2: Reject empty or whitespace-only question with 4xx status."""
    with open("fixtures/image_a.jpg", "rb") as img:
        files = {"file": ("image_a.jpg", img.read(), "image/jpeg")}

    # Empty string question
    res_empty = client.post("/api/vqa", files=files, data={"question": ""})
    assert res_empty.status_code in (400, 422), f"Expected 4xx, got {res_empty.status_code}"

    # Whitespace question
    with open("fixtures/image_a.jpg", "rb") as img:
        files = {"file": ("image_a.jpg", img.read(), "image/jpeg")}
    res_ws = client.post("/api/vqa", files=files, data={"question": "   \n\t  "})
    assert res_ws.status_code in (400, 422), f"Expected 4xx, got {res_ws.status_code}"

    # Omitted question
    with open("fixtures/image_a.jpg", "rb") as img:
        files = {"file": ("image_a.jpg", img.read(), "image/jpeg")}
    res_omit = client.post("/api/vqa", files=files)
    assert res_omit.status_code in (400, 422), f"Expected 4xx, got {res_omit.status_code}"


def test_req3_validation_missing_or_invalid_image(client):
    """Requirement 3: Reject missing, non-image, or empty files with 4xx status."""
    # Omitted file
    res_omit = client.post("/api/vqa", data={"question": "How many apples?"})
    assert res_omit.status_code in (400, 422), f"Expected 4xx, got {res_omit.status_code}"

    # Text file masquerading as image or invalid type
    res_txt = client.post(
        "/api/vqa",
        files={"file": ("test.txt", b"plain text content", "text/plain")},
        data={"question": "What is this?"}
    )
    assert res_txt.status_code in (400, 422), f"Expected 4xx, got {res_txt.status_code}"

    # Empty file
    res_empty = client.post(
        "/api/vqa",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
        data={"question": "What is this?"}
    )
    assert res_empty.status_code in (400, 422), f"Expected 4xx, got {res_empty.status_code}"

    # Corrupted image bytes
    res_corrupt = client.post(
        "/api/vqa",
        files={"file": ("corrupt.jpg", b"not-a-valid-image-bytes", "image/jpeg")},
        data={"question": "What is this?"}
    )
    assert res_corrupt.status_code in (400, 422), f"Expected 4xx, got {res_corrupt.status_code}"


def test_req4_and_req5_grounding_proof(client):
    """Requirement 4 & 5: Successful response format and visual grounding verification."""
    with open("fixtures/fixtures.json", "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    question = fixtures["question"]

    # Image A request
    with open(fixtures["image_a_path"], "rb") as f_a:
        res_a = client.post(
            "/api/vqa",
            files={"file": (Path(fixtures["image_a_path"]).name, f_a.read(), "image/jpeg")},
            data={"question": question}
        )
    assert res_a.status_code == 200, f"Expected 200, got {res_a.status_code}: {res_a.text}"
    json_a = res_a.json()
    assert "answer" in json_a and isinstance(json_a["answer"], str)
    assert fixtures["expected_a_contains"].lower() in json_a["answer"].lower(), (
        f"Expected '{fixtures['expected_a_contains']}' in answer '{json_a['answer']}'"
    )

    # Image B request
    with open(fixtures["image_b_path"], "rb") as f_b:
        res_b = client.post(
            "/api/vqa",
            files={"file": (Path(fixtures["image_b_path"]).name, f_b.read(), "image/jpeg")},
            data={"question": question}
        )
    assert res_b.status_code == 200, f"Expected 200, got {res_b.status_code}: {res_b.text}"
    json_b = res_b.json()
    assert "answer" in json_b and isinstance(json_b["answer"], str)
    assert fixtures["expected_b_contains"].lower() in json_b["answer"].lower(), (
        f"Expected '{fixtures['expected_b_contains']}' in answer '{json_b['answer']}'"
    )

    # Grounding proof: answers must differ
    assert json_a["answer"] != json_b["answer"], "Answers must functionally differ based on visual input"


def test_req6_cryptographic_audit_log(client):
    """Requirement 6: SHA-256 hash written to audit.log upon 200 OK."""
    with open("fixtures/fixtures.json", "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    with open(fixtures["image_a_path"], "rb") as f:
        image_bytes = f.read()

    expected_hash = hashlib.sha256(image_bytes).hexdigest()

    res = client.post(
        "/api/vqa",
        files={"file": ("image_a.jpg", image_bytes, "image/jpeg")},
        data={"question": fixtures["question"]}
    )
    assert res.status_code == 200

    audit_log = Path("audit.log")
    assert audit_log.is_file(), "audit.log must be created at runtime"
    with open(audit_log, "r", encoding="utf-8") as f:
        log_content = f.read()

    assert expected_hash in log_content, f"Expected SHA-256 hash {expected_hash} in audit.log"


def test_req7_health_check(client):
    """Requirement 7: Health check endpoint confirms API readiness."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert data["status"] in ("healthy", "ok")


def test_req8_docker_compose_file():
    """Requirement 8: docker-compose.yml configuration validation."""
    dc_path = Path("docker-compose.yml")
    assert dc_path.is_file(), "docker-compose.yml must exist"
    content = dc_path.read_text(encoding="utf-8")
    assert "8000:8000" in content, "Must map port 8000 to host"
    assert "healthcheck" in content, "Must configure container healthcheck"
    assert "/health" in content, "Healthcheck must target /health"
    assert "env_file" in content, "Must configure env_file"


def test_req9_env_example():
    """Requirement 9: .env.example contains documented placeholders and no secrets."""
    env_ex = Path(".env.example")
    assert env_ex.is_file(), ".env.example must exist"
    lines = [line.strip() for line in env_ex.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    keys = [line.split("=", 1)[0] for line in lines if "=" in line]
    assert "OPENAI_API_KEY" in keys or "ANTHROPIC_API_KEY" in keys, "Must include VLM API key placeholders"
    # Ensure no actual sk- production secrets committed
    assert "sk-" not in env_ex.read_text(encoding="utf-8")


def test_req10_submission_json():
    """Requirement 10: submission.json configuration validation."""
    sub_path = Path("submission.json")
    assert sub_path.is_file(), "submission.json must exist"
    with open(sub_path, "r", encoding="utf-8") as f:
        sub = json.load(f)
    assert "vlm_provider" in sub and isinstance(sub["vlm_provider"], str)
    assert "model_name" in sub and isinstance(sub["model_name"], str)

