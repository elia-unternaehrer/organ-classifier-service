"""Scaffolding tests.

These exist so that CI has something real to assert from the first commit
onward. They are replaced by the actual preprocessing, parity and API tests as
those parts land.
"""

import tomllib
from pathlib import Path

import organ_service

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_and_exposes_version() -> None:
    assert organ_service.__version__


def test_version_matches_pyproject() -> None:
    """The package version and the project metadata must not drift apart.

    The serving layer reports its version on /health, and release tags are cut
    from it, so a mismatch here would make deployed builds unidentifiable.
    """
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    assert pyproject["project"]["version"] == organ_service.__version__


def test_runtime_dependencies_exclude_torch() -> None:
    """Torch must never reach the serving image.

    Inference runs on ONNX Runtime. This test guards the boundary: if torch is
    ever added to the runtime dependency list, the build breaks here rather
    than silently shipping a multi-gigabyte image.
    """
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    runtime = [dep.lower() for dep in pyproject["project"]["dependencies"]]
    assert not any(dep.startswith(("torch", "timm", "medmnist")) for dep in runtime)
