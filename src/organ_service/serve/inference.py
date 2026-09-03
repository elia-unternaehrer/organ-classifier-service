"""ONNX inference, and the registry of models the service exposes.

Deliberately free of FastAPI. Everything the service actually does happens
here, so it can be tested by calling functions rather than by driving HTTP,
and the app layer stays thin enough to read in one sitting.

The registry exists because the interesting claim of this project is a
comparison. A demo that serves one model asks the visitor to believe that
quantisation costs nothing; a demo that serves several lets them switch and
watch the prediction stay the same while the latency moves. Holding more than
one session is the difference between asserting the result and showing it.

Each model configures itself from its own metadata, so the registry can hold
artefacts at different resolutions and precisions without the service knowing
anything about them beyond a file path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from organ_service.model_meta import ModelMetadata
from organ_service.preprocessing import add_batch_axis, preprocess

DEFAULT_INTRA_OP_THREADS = 1
"""One thread by default.

Sized for the deployment target rather than for a workstation. On a shared-CPU
dyno the extra threads contend rather than help, and each one costs memory the
512 MB quota does not have. Raise it via configuration when running on real
hardware.
"""


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution settings for every session in the registry.

    Attributes:
        intra_op_threads: Threads per operator.
        enable_memory_arena: ONNX Runtime's arena allocator preallocates
            generously and does not return memory between requests. Off by
            default because the deployment target restarts the process when it
            exceeds its quota, which is a worse outcome than slightly slower
            allocation.
    """

    intra_op_threads: int = DEFAULT_INTRA_OP_THREADS
    enable_memory_arena: bool = False

    def to_session_options(self) -> ort.SessionOptions:
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.intra_op_threads
        options.enable_cpu_mem_arena = self.enable_memory_arena
        return options


@dataclass(frozen=True)
class Prediction:
    class_index: int
    class_name: str
    probability: float


@dataclass(frozen=True)
class InferenceResult:
    """One prediction, with the timing split that motivated the project.

    Preprocessing and inference are reported separately because they scale
    differently with resolution: decoding and resampling cost roughly the same
    whether the target is 64 or 224 pixels, while the forward pass scales with
    the pixel count. A model that is ten times cheaper to run is not ten times
    cheaper to serve, and separating the two is the only way to see that.
    """

    model_id: str
    predictions: list[Prediction]
    preprocess_ms: float
    inference_ms: float

    @property
    def top(self) -> Prediction:
        return self.predictions[0]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis.

    Applied here rather than baked into the graph. The exported model ends at
    logits, which is what the parity check compares and what the quantisation
    tools expect; adding a softmax node would change that contract for the sake
    of three lines of NumPy.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


class LoadedModel:
    """One ONNX session plus the metadata that describes how to feed it."""

    def __init__(self, path: Path, runtime: RuntimeConfig | None = None) -> None:
        self.path = path
        runtime = runtime or RuntimeConfig()

        self.session = ort.InferenceSession(
            str(path),
            sess_options=runtime.to_session_options(),
            providers=["CPUExecutionProvider"],
        )
        self.metadata = ModelMetadata.from_props(self.session.get_modelmeta().custom_metadata_map)

    @property
    def model_id(self) -> str:
        """Stable identifier used in the API and the frontend dropdown.

        Built from run name and precision rather than from the filename, so it
        stays correct if the file is renamed, and so it distinguishes the same
        precision at different resolutions.
        """
        return f"{self.metadata.run_name}_{self.metadata.precision}"

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    def predict(self, image: Image.Image) -> InferenceResult:
        """Run one image through the model.

        Args:
            image: An already decoded PIL image, in any mode or size.

        Returns:
            All classes, sorted by descending probability. The caller decides
            how many to display; truncating here would make the response
            unusable for anything but the demo.
        """
        started = time.perf_counter()
        sample = preprocess(image, size=self.metadata.image_size)
        batch = add_batch_axis(sample).astype(np.float32)
        preprocess_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        logits = self.session.run(["logits"], {"input": batch})[0]
        inference_ms = (time.perf_counter() - started) * 1000

        probabilities = softmax(logits[0])
        order = np.argsort(probabilities)[::-1]

        return InferenceResult(
            model_id=self.model_id,
            predictions=[
                Prediction(
                    class_index=int(index),
                    class_name=self.metadata.class_names[index],
                    probability=float(probabilities[index]),
                )
                for index in order
            ],
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
        )


class ModelRegistry:
    """The set of models the service will serve.

    Sessions are created eagerly at startup. Lazy loading would make the first
    request to each model slow and, more importantly, would make peak memory
    depend on which models a visitor happened to click: on a platform that
    restarts the process when it exceeds its quota, memory use needs to be
    known at startup, not discovered under traffic.
    """

    def __init__(self, models: list[LoadedModel], default_id: str | None = None) -> None:
        if not models:
            raise ValueError("registry needs at least one model")

        self._models = {model.model_id: model for model in models}

        if len(self._models) != len(models):
            raise ValueError(
                "two artefacts share a model id; each must have a distinct "
                "combination of run name and precision"
            )

        # A dropdown that switches between models labelled with different class
        # names would be nonsense, and the mismatch would be invisible until a
        # visitor compared two predictions.
        reference = models[0].metadata.class_names
        for model in models[1:]:
            if model.metadata.class_names != reference:
                raise ValueError(
                    f"{model.model_id} has different class names from "
                    f"{models[0].model_id}; the registry is meant to hold "
                    f"variants of one model, not unrelated ones"
                )

        if default_id is not None and default_id not in self._models:
            raise ValueError(f"default model {default_id!r} is not among {sorted(self._models)}")
        self._default_id = default_id or models[0].model_id

    @classmethod
    def from_paths(
        cls,
        paths: list[Path],
        default_id: str | None = None,
        runtime: RuntimeConfig | None = None,
    ) -> ModelRegistry:
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("no artefact at " + ", ".join(str(path) for path in missing))
        return cls([LoadedModel(path, runtime) for path in paths], default_id)

    @property
    def default_id(self) -> str:
        return self._default_id

    @property
    def ids(self) -> list[str]:
        return sorted(self._models)

    @property
    def class_names(self) -> list[str]:
        return self._models[self._default_id].metadata.class_names

    def get(self, model_id: str | None = None) -> LoadedModel:
        """Look up a model, falling back to the default.

        Raises:
            KeyError: With the available ids listed, since an API client that
                guessed wrong needs to know what it may ask for.
        """
        if model_id is None:
            return self._models[self._default_id]
        if model_id not in self._models:
            raise KeyError(f"unknown model {model_id!r}; available: {self.ids}")
        return self._models[model_id]

    def describe(self) -> list[dict]:
        """Summarise the registry for /health and the frontend dropdown."""
        return [
            {
                "model_id": model.model_id,
                "run_name": model.metadata.run_name,
                "precision": model.metadata.precision,
                "image_size": model.metadata.image_size,
                "model_version": model.metadata.model_version,
                "git_sha": model.metadata.git_sha,
                "size_bytes": model.size_bytes,
                "is_default": model.model_id == self._default_id,
            }
            for model in (self._models[key] for key in self.ids)
        ]
