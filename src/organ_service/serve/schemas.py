"""Response schemas.

Explicit models rather than bare dicts, so the OpenAPI document at /docs is a
usable description of the API rather than a list of endpoints that return
"object". Request bodies are multipart uploads and are declared inline in the
route.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionOut(BaseModel):
    class_index: int = Field(description="Index into the model's class list")
    class_name: str
    probability: float = Field(ge=0.0, le=1.0)


class ModelInfo(BaseModel):
    """One artefact in the registry.

    Read out of the model's own metadata, not out of service configuration, so
    what is reported here is necessarily what is loaded.
    """

    model_id: str
    run_name: str
    precision: str
    image_size: int
    model_version: str
    git_sha: str
    size_bytes: int
    is_default: bool

    # 'model_' is Pydantic's protected prefix; these fields describe an ML
    # model rather than a Pydantic one, so the protection is switched off
    # rather than the names being bent around it.
    model_config = {"protected_namespaces": ()}


class Timing(BaseModel):
    """Milliseconds, split by stage.

    Reported per request because the two stages scale differently with input
    resolution: decode and resample cost about the same at any target size,
    while the forward pass scales with pixel count. A model that is ten times
    cheaper to run is not ten times cheaper to serve, and only the split shows
    that.
    """

    preprocess_ms: float
    inference_ms: float
    total_ms: float


class PredictResponse(BaseModel):
    model_id: str
    predictions: list[PredictionOut]
    timing: Timing

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    status: str
    service_version: str
    models: list[ModelInfo]
    default_model_id: str
    class_names: list[str]


class ReadyResponse(BaseModel):
    ready: bool
    models_loaded: int


class ErrorResponse(BaseModel):
    detail: str
