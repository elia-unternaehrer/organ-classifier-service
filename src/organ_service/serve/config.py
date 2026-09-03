"""Service configuration, read from the environment.

Plain dataclass rather than a settings library. The service has six knobs, all
of them strings or numbers, and the validation that matters is not type
checking but the bounds below: a thread count that fits the deployment target
and an upload limit that fits its memory.

Which models to serve is configuration, not code. The same image runs with one
model on a small dyno and with four on a machine that has the memory, and the
difference is one environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_DIR = Path("artifacts")
DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _split_paths(raw: str) -> list[Path]:
    return [Path(part.strip()) for part in raw.split(",") if part.strip()]


@dataclass(frozen=True)
class ServiceConfig:
    """Everything the service reads at startup.

    Attributes:
        model_paths: Artefacts to load, in order. The first is the default
            unless ``default_model_id`` says otherwise.
        default_model_id: Which model answers requests that do not name one.
        max_upload_bytes: Uploads beyond this are rejected with 413 before
            being read into memory.
        intra_op_threads: Threads per ONNX operator. One by default, because
            the deployment target has a fraction of a CPU and extra threads
            contend rather than help.
        enable_memory_arena: ONNX Runtime's arena allocator, off by default.
            It preallocates generously and does not release between requests,
            which on a memory-capped platform trades a small speedup for
            process restarts.
    """

    model_paths: list[Path] = field(default_factory=list)
    default_model_id: str | None = None
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    intra_op_threads: int = 1
    enable_memory_arena: bool = False

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> ServiceConfig:
        """Build from environment variables.

        ``MODEL_PATHS`` is a comma-separated list. When it is absent, every
        ``.onnx`` file in ``MODEL_DIR`` is loaded in sorted order, which makes
        local development work with no configuration at all: export, quantise,
        run.

        Raises:
            ValueError: If no artefacts can be found. Starting a model server
                with no model is never what anyone meant.
        """
        env = os.environ if environ is None else environ

        raw_paths = env.get("MODEL_PATHS", "").strip()
        if raw_paths:
            paths = _split_paths(raw_paths)
        else:
            model_dir = Path(env.get("MODEL_DIR", str(DEFAULT_MODEL_DIR)))
            paths = sorted(model_dir.glob("*.onnx"))
            if not paths:
                raise ValueError(
                    f"no .onnx artefacts in {model_dir}. Set MODEL_PATHS "
                    f"explicitly, or run 'organ-service export' first."
                )

        return cls(
            model_paths=paths,
            default_model_id=env.get("DEFAULT_MODEL") or None,
            max_upload_bytes=int(env.get("MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)),
            intra_op_threads=int(env.get("INTRA_OP_THREADS", "1")),
            enable_memory_arena=env.get("ENABLE_MEMORY_ARENA", "").lower() in {"1", "true", "yes"},
        )
