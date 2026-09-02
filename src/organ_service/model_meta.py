"""Metadata carried inside the ONNX artefact.

The exported model describes itself. Input resolution, normalisation constants
and class names travel in the graph's ``metadata_props``, so the service reads
its configuration out of whichever artefact it was handed rather than from a
config file that could disagree with it.

That disagreement is the failure this prevents. Serving a model exported at
224 with a service configured for 64 produces no error, no warning and no
crash. It produces predictions that are quietly wrong, which is the worst
category of production bug.

Writing requires the ``onnx`` package and happens at export time. Reading goes
through ``onnxruntime``'s ``get_modelmeta()``, which exposes the same fields
without ``onnx`` installed, so the serving image needs neither torch nor onnx.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

SCHEMA_VERSION = "1"

# Prefixed to avoid colliding with keys any tooling may add to the same map.
KEY_PREFIX = "organ_service."


@dataclass(frozen=True)
class ModelMetadata:
    """Everything needed to serve a model correctly.

    Attributes:
        schema_version: Bumped when the key set changes, so an old service
            refuses a newer artefact instead of misreading it.
        model_version: The release tag the artefact was published under.
        run_name: The training run that produced it, e.g. ``resnet18_224``.
        precision: ``fp32``, ``int8_dynamic`` or ``int8_static``.
        image_size: Square input side length the graph expects.
        norm_mean: Normalisation constant applied after scaling to [0, 1].
        norm_std: Normalisation constant applied after scaling to [0, 1].
        class_names: Ordered so that index equals the logit index.
        git_sha: Commit that produced the artefact.
        package_version: Version of this package at export time.
    """

    schema_version: str
    model_version: str
    run_name: str
    precision: str
    image_size: int
    norm_mean: float
    norm_std: float
    class_names: list[str]
    git_sha: str
    package_version: str

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def to_props(self) -> dict[str, str]:
        """Flatten to the string-to-string map ONNX metadata allows."""
        return {
            f"{KEY_PREFIX}schema_version": self.schema_version,
            f"{KEY_PREFIX}model_version": self.model_version,
            f"{KEY_PREFIX}run_name": self.run_name,
            f"{KEY_PREFIX}precision": self.precision,
            f"{KEY_PREFIX}image_size": str(self.image_size),
            f"{KEY_PREFIX}norm_mean": repr(self.norm_mean),
            f"{KEY_PREFIX}norm_std": repr(self.norm_std),
            f"{KEY_PREFIX}class_names": json.dumps(self.class_names),
            f"{KEY_PREFIX}git_sha": self.git_sha,
            f"{KEY_PREFIX}package_version": self.package_version,
        }

    @classmethod
    def from_props(cls, props: dict[str, str]) -> ModelMetadata:
        """Rebuild from ONNX Runtime's ``custom_metadata_map``.

        Raises:
            ValueError: If keys are missing or the schema version is one this
                code does not know. Both are refusals by design: a model
                without readable preprocessing parameters must not be served
                on guessed defaults.
        """
        missing = [
            key
            for key in (
                "schema_version",
                "model_version",
                "run_name",
                "precision",
                "image_size",
                "norm_mean",
                "norm_std",
                "class_names",
            )
            if f"{KEY_PREFIX}{key}" not in props
        ]
        if missing:
            raise ValueError(
                "artefact is missing metadata keys: "
                + ", ".join(missing)
                + ". It was probably exported by an older version of "
                "scripts/export_onnx.py; re-export it."
            )

        version = props[f"{KEY_PREFIX}schema_version"]
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"artefact declares metadata schema {version}, this build "
                f"understands {SCHEMA_VERSION}"
            )

        return cls(
            schema_version=version,
            model_version=props[f"{KEY_PREFIX}model_version"],
            run_name=props[f"{KEY_PREFIX}run_name"],
            precision=props[f"{KEY_PREFIX}precision"],
            image_size=int(props[f"{KEY_PREFIX}image_size"]),
            norm_mean=float(props[f"{KEY_PREFIX}norm_mean"]),
            norm_std=float(props[f"{KEY_PREFIX}norm_std"]),
            class_names=json.loads(props[f"{KEY_PREFIX}class_names"]),
            git_sha=props.get(f"{KEY_PREFIX}git_sha", "unknown"),
            package_version=props.get(f"{KEY_PREFIX}package_version", "unknown"),
        )

    def summary(self) -> str:
        """One line for startup logs and /health."""
        return (
            f"{self.run_name} {self.precision} "
            f"({self.image_size}px, {self.num_classes} classes, "
            f"version {self.model_version})"
        )
