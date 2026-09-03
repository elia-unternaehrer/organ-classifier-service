"""Tests for the HTTP layer.

Driven through the real ASGI stack rather than by calling handlers, so route
declarations, response models and status codes are all exercised. The registry
is injected, so no artefact is read from disk and these run in the default CI
job.
"""

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from organ_service.serve.app import create_app
from organ_service.serve.config import ServiceConfig
from organ_service.serve.inference import ModelRegistry
from tests.test_inference import build_model


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry.from_paths(
        [
            build_model(tmp_path / "fp32.onnx", "resnet18_224", "fp32", 224),
            build_model(tmp_path / "int8.onnx", "resnet18_224", "int8_static", 224),
            build_model(tmp_path / "small.onnx", "resnet18_64", "fp32", 64),
        ]
    )


# Low enough that the oversized case is cheap to trigger, high enough that a
# realistic slice upload passes. Random-noise PNGs barely compress, so a 32x32
# fixture is already several kilobytes.
MAX_UPLOAD_BYTES = 200_000


@pytest.fixture
def client(registry: ModelRegistry) -> TestClient:
    app = create_app(config=ServiceConfig(max_upload_bytes=MAX_UPLOAD_BYTES), registry=registry)
    with TestClient(app) as test_client:
        yield test_client


def png_bytes(size: tuple[int, int] = (32, 32)) -> bytes:
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (size[1], size[0]), dtype=np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def upload(client: TestClient, data: bytes, **params) -> object:
    return client.post(
        "/predict",
        files={"file": ("slice.png", data, "image/png")},
        params=params,
    )


# --- happy path ------------------------------------------------------------


def test_predict_returns_every_class(client: TestClient) -> None:
    response = upload(client, png_bytes())
    assert response.status_code == 200

    body = response.json()
    assert body["model_id"] == "resnet18_224_fp32"
    assert len(body["predictions"]) == 4
    assert body["timing"]["total_ms"] > 0


def test_predict_probabilities_sum_to_one(client: TestClient) -> None:
    body = upload(client, png_bytes()).json()
    total = sum(p["probability"] for p in body["predictions"])
    assert total == pytest.approx(1.0)


def test_top_k_truncates(client: TestClient) -> None:
    body = upload(client, png_bytes(), top_k=2).json()
    assert len(body["predictions"]) == 2


def test_model_can_be_selected(client: TestClient) -> None:
    """The query parameter behind the frontend dropdown."""
    body = upload(client, png_bytes(), model="resnet18_224_int8_static").json()
    assert body["model_id"] == "resnet18_224_int8_static"


def test_models_at_different_resolutions_both_answer(client: TestClient) -> None:
    """Switching resolution needs no change in the request beyond the id."""
    for model_id in ("resnet18_224_fp32", "resnet18_64_fp32"):
        assert upload(client, png_bytes(), model=model_id).json()["model_id"] == model_id


def test_realistic_upload_size_is_accepted(client: TestClient) -> None:
    """Browser uploads are whatever the visitor had to hand, not 28x28."""
    payload = png_bytes((200, 150))
    assert len(payload) < MAX_UPLOAD_BYTES
    assert upload(client, payload).status_code == 200


# --- error handling --------------------------------------------------------


def test_unknown_model_is_404_and_lists_alternatives(client: TestClient) -> None:
    response = upload(client, png_bytes(), model="resnet18_224_int8_dynamic")

    assert response.status_code == 404
    assert "available" in response.json()["detail"]


def test_undecodable_upload_is_400_not_500(client: TestClient) -> None:
    """A corrupt file is the client's problem, and must not read as an outage."""
    response = upload(client, b"this is not an image")

    assert response.status_code == 400
    assert "decode" in response.json()["detail"]


def test_oversized_upload_is_413(client: TestClient) -> None:
    """Rejected on size before the decoder ever sees it."""
    response = upload(client, b"\x89PNG" + b"0" * (MAX_UPLOAD_BYTES + 1))
    assert response.status_code == 413


def test_empty_upload_is_400(client: TestClient) -> None:
    assert upload(client, b"").status_code == 400


def test_non_image_content_type_is_rejected(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_missing_file_is_422(client: TestClient) -> None:
    """FastAPI's own validation; asserted so the contract is pinned."""
    assert client.post("/predict").status_code == 422


# --- introspection ---------------------------------------------------------


def test_health_reports_what_is_loaded(client: TestClient) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["default_model_id"] == "resnet18_224_fp32"
    assert len(body["models"]) == 3
    assert body["class_names"][0] == "liver"


def test_health_exposes_provenance(client: TestClient) -> None:
    """Which build is running is the question asked during an incident."""
    entry = client.get("/health").json()["models"][0]

    assert entry["git_sha"] == "abc123"
    assert entry["model_version"] == "v0.1.0"
    assert entry["image_size"] == 224


def test_health_marks_exactly_one_default(client: TestClient) -> None:
    models = client.get("/health").json()["models"]
    assert sum(entry["is_default"] for entry in models) == 1


def test_ready(client: TestClient) -> None:
    body = client.get("/ready").json()
    assert body["ready"] is True
    assert body["models_loaded"] == 3


def test_metrics_are_prometheus_formatted(client: TestClient) -> None:
    upload(client, png_bytes())
    body = client.get("/metrics").text

    assert "organ_predictions_total" in body
    assert "organ_prediction_stage_seconds" in body


def test_metrics_count_per_model_and_outcome(client: TestClient) -> None:
    """Failures are labelled by cause, not lumped into one error counter."""
    upload(client, png_bytes())
    upload(client, b"not an image")
    body = client.get("/metrics").text

    assert 'outcome="success"' in body
    assert 'outcome="undecodable"' in body


def test_metrics_expose_loaded_model_versions(client: TestClient) -> None:
    body = client.get("/metrics").text
    assert "organ_model_loaded" in body
    assert 'precision="int8_static"' in body


def test_openapi_document_builds(client: TestClient) -> None:
    """Guards the response models; a bad schema fails here, not at /docs."""
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]


def test_root_responds_without_a_frontend(client: TestClient) -> None:
    """The API is complete on its own, and the image builds before the demo."""
    assert client.get("/").status_code == 200
