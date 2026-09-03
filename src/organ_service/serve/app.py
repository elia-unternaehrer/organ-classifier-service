"""The HTTP layer.

Thin by design. Everything the service actually computes lives in
``inference``, which is why that module can be tested by calling functions;
what happens here is request validation, error mapping and instrumentation.

    uvicorn organ_service.serve.app:app --port 8000

The application is built by a factory so that tests can inject a registry
instead of loading artefacts from disk, and so that a different model set is a
configuration change rather than a code change.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from organ_service import __version__
from organ_service.preprocessing import load_image
from organ_service.serve.config import ServiceConfig
from organ_service.serve.inference import ModelRegistry, RuntimeConfig
from organ_service.serve.schemas import (
    HealthResponse,
    PredictResponse,
    ReadyResponse,
)
from organ_service.serve.telemetry import Telemetry

STATIC_DIR = Path(__file__).parent / "static"

ACCEPTED_PREFIXES = ("image/",)


def build_registry(config: ServiceConfig) -> ModelRegistry:
    return ModelRegistry.from_paths(
        config.model_paths,
        default_id=config.default_model_id,
        runtime=RuntimeConfig(
            intra_op_threads=config.intra_op_threads,
            enable_memory_arena=config.enable_memory_arena,
        ),
    )


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_telemetry(request: Request) -> Telemetry:
    return request.app.state.telemetry


def get_config(request: Request) -> ServiceConfig:
    return request.app.state.config


def create_app(
    config: ServiceConfig | None = None,
    registry: ModelRegistry | None = None,
) -> FastAPI:
    """Assemble the application.

    Args:
        config: Read from the environment when omitted.
        registry: Injected by tests. When given, no artefacts are loaded from
            disk and ``config.model_paths`` is ignored.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Models load here, before the port accepts traffic. Failing at startup
        # is the right outcome: a model server that starts without a model
        # would answer health checks while being useless.
        app.state.config = config or ServiceConfig.from_env()
        app.state.registry = registry or build_registry(app.state.config)
        app.state.telemetry = Telemetry()
        app.state.telemetry.record_models(app.state.registry.describe())
        yield

    app = FastAPI(
        title="Organ Classifier Service",
        description=(
            "Organ classification on abdominal CT slices. Several artefacts "
            "are served side by side so that precision and resolution can be "
            "compared at request time."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(
        file: UploadFile = File(description="Image to classify"),
        model: str | None = Query(
            default=None,
            description="Model id from /health; the default is used when omitted",
        ),
        top_k: int | None = Query(
            default=None, ge=1, description="Return only the k most likely classes"
        ),
        registry: ModelRegistry = Depends(get_registry),
        telemetry: Telemetry = Depends(get_telemetry),
        config: ServiceConfig = Depends(get_config),
    ) -> PredictResponse:
        try:
            selected = registry.get(model)
        except KeyError as exc:
            telemetry.record_failure(model or "unknown", "unknown_model")
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if file.content_type and not file.content_type.startswith(ACCEPTED_PREFIXES):
            telemetry.record_failure(selected.model_id, "wrong_type")
            raise HTTPException(
                status_code=400,
                detail=f"expected an image, got content type {file.content_type!r}",
            )

        # One byte past the limit is enough to know it was exceeded, and stops
        # an oversized upload from being read into memory in full.
        payload = await file.read(config.max_upload_bytes + 1)
        if len(payload) > config.max_upload_bytes:
            telemetry.record_failure(selected.model_id, "too_large")
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds {config.max_upload_bytes} bytes",
            )
        if not payload:
            telemetry.record_failure(selected.model_id, "empty")
            raise HTTPException(status_code=400, detail="uploaded file is empty")

        try:
            image = load_image(payload)
        except ValueError as exc:
            # A file the decoder rejects is a bad request, not a server fault.
            telemetry.record_failure(selected.model_id, "undecodable")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = selected.predict(image)
        telemetry.record_success(
            result.model_id,
            result.preprocess_ms / 1000,
            result.inference_ms / 1000,
        )

        predictions = result.predictions[:top_k] if top_k else result.predictions
        return PredictResponse(
            model_id=result.model_id,
            predictions=[p.__dict__ for p in predictions],
            timing={
                "preprocess_ms": round(result.preprocess_ms, 2),
                "inference_ms": round(result.inference_ms, 2),
                "total_ms": round(result.preprocess_ms + result.inference_ms, 2),
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def health(
        registry: ModelRegistry = Depends(get_registry),
    ) -> HealthResponse:
        """What is actually loaded, read from the artefacts themselves."""
        return HealthResponse(
            status="ok",
            service_version=__version__,
            models=registry.describe(),
            default_model_id=registry.default_id,
            class_names=registry.class_names,
        )

    @app.get("/ready", response_model=ReadyResponse)
    async def ready(registry: ModelRegistry = Depends(get_registry)) -> ReadyResponse:
        """Readiness for platform health checks.

        Reached only after the lifespan handler has loaded every session, so
        answering at all is the signal.
        """
        return ReadyResponse(ready=True, models_loaded=len(registry.ids))

    @app.get("/metrics")
    async def metrics(telemetry: Telemetry = Depends(get_telemetry)) -> Response:
        return PlainTextResponse(
            generate_latest(telemetry.registry), media_type=CONTENT_TYPE_LATEST
        )

    # The browser demo is optional: the API is complete without it, and the
    # image builds before the frontend exists.
    if (STATIC_DIR / "index.html").exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        async def index_placeholder() -> dict:
            return {
                "service": "organ-classifier-service",
                "version": __version__,
                "docs": "/docs",
                "note": "no frontend bundled in this build",
            }

    return app


app = create_app
"""Lazily bound so importing this module does not load artefacts.

``uvicorn organ_service.serve.app:app`` calls it, which is what ASGI servers do
with a callable. Tests import ``create_app`` directly.
"""
