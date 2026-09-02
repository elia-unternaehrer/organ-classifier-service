"""Loading and provenance for the MedMNIST arrays.

Deliberately free of torch. Everything here is plain NumPy so that the loading
logic, the split bookkeeping and the provenance record can be tested in CI,
where only the runtime and dev dependency groups are installed. The torch
``Dataset`` that consumes these arrays lives in ``organ_service.dataset``.

The provenance record is the substitute for a data-versioning tool. It answers
the question DVC would answer, which bytes produced these numbers, by carrying
the archive hash and split sizes into ``metrics.json`` alongside the results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Split:
    """One split's images and labels, still as raw uint8.

    Kept untransformed on purpose: the transform is applied per sample so that
    augmentation can vary between epochs, and holding 34k pre-transformed
    224-pixel float tensors would not fit in memory anyway.

    Attributes:
        images: ``(N, H, W)`` uint8.
        labels: ``(N,)`` int64, already squeezed from MedMNIST's ``(N, 1)``.
    """

    images: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return int(self.images.shape[0])


@dataclass(frozen=True)
class DatasetProvenance:
    """Identifies the exact data artefact behind a training run."""

    dataset: str
    size: int
    npz_filename: str
    medmnist_version: str
    expected_md5: str
    file_sha256: str
    file_bytes: int
    split_sizes: dict[str, int]
    class_names: list[str]
    license: str
    source_url: str
    split_note: str

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def to_dict(self) -> dict:
        return asdict(self)


def npz_filename(dataset: str, size: int) -> str:
    """Reproduce MedMNIST's on-disk naming; 28 carries no size suffix."""
    suffix = "" if size == 28 else f"_{size}"
    return f"{dataset}{suffix}.npz"


def load_manifest(dataset: str, size: int, root: Path) -> DatasetProvenance:
    """Read the sidecar written by ``scripts/download_data.py``.

    Raises:
        FileNotFoundError: With the exact command needed to produce it. A
            missing manifest is the most likely first-run failure, so the
            message is the fix rather than a description of the problem.
    """
    path = (root / npz_filename(dataset, size)).with_suffix(".manifest.json")
    if not path.exists():
        raise FileNotFoundError(
            f"no manifest at {path}. Run:\n"
            f"  uv run --extra train python scripts/download_data.py "
            f"--dataset {dataset} --size {size} --root {root}"
        )

    raw = json.loads(path.read_text())
    fields = {f for f in DatasetProvenance.__dataclass_fields__}
    return DatasetProvenance(**{k: v for k, v in raw.items() if k in fields})


def load_split(dataset: str, size: int, root: Path, split: str, mmap: bool = False) -> Split:
    """Load one split from the archive.

    Args:
        mmap: Memory-map instead of reading into RAM. The 224-pixel training
            split is roughly 1.7 GB as uint8, so this is the escape hatch on
            machines where that does not fit comfortably. Off by default
            because resident arrays are meaningfully faster to sample from.

    Raises:
        ValueError: If ``split`` is not one of the official three. Custom
            splits are not supported by design: resplitting would break
            comparability with the published baselines and, worse, would draw
            at slice level and put the same patient on both sides.
    """
    if split not in SPLITS:
        raise ValueError(f"expected one of {SPLITS}, got {split!r}")

    path = root / npz_filename(dataset, size)
    if not path.exists():
        raise FileNotFoundError(
            f"no archive at {path}. Run:\n"
            f"  uv run --extra train python scripts/download_data.py "
            f"--dataset {dataset} --size {size} --root {root}"
        )

    with np.load(path, mmap_mode="r" if mmap else None) as archive:
        images = archive[f"{split}_images"]
        labels = archive[f"{split}_labels"]
        if not mmap:
            images = np.asarray(images)
            labels = np.asarray(labels)

    # MedMNIST stores labels as (N, 1) for single-label tasks.
    labels = np.asarray(labels).reshape(-1).astype(np.int64)

    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"{split}: {images.shape[0]} images but {labels.shape[0]} labels")

    return Split(images=images, labels=labels)


def class_distribution(split: Split, num_classes: int) -> list[int]:
    """Count samples per class.

    Reported per split in ``metrics.json``. Balanced accuracy is the selection
    metric precisely because these counts are not uniform, so publishing the
    counts alongside the metric lets a reader judge it.
    """
    return np.bincount(split.labels, minlength=num_classes).tolist()
