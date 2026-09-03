"""Tests for the torch-free data layer.

A synthetic archive stands in for the real download: it mirrors MedMNIST's key
naming and its ``(N, 1)`` label shape, which is enough to exercise every code
path here without a multi-gigabyte fixture. CI installs no torch, so these run
in the default job.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from organ_service import augment
from organ_service import data as dm
from organ_service.preprocessing import IDENTITY

NUM_CLASSES = 11
SIZE = 28


@pytest.fixture
def archive_root(tmp_path: Path) -> Path:
    """Write a miniature archive plus manifest in MedMNIST's own layout."""
    rng = np.random.default_rng(0)
    counts = {"train": 40, "val": 12, "test": 20}

    arrays = {}
    for split, n in counts.items():
        arrays[f"{split}_images"] = rng.integers(0, 256, (n, SIZE, SIZE), dtype=np.uint8)
        # MedMNIST stores single-label targets with a trailing axis.
        arrays[f"{split}_labels"] = rng.integers(0, NUM_CLASSES, (n, 1), dtype=np.uint8)

    npz_path = tmp_path / "organamnist.npz"
    np.savez(npz_path, **arrays)

    manifest = {
        "dataset": "organamnist",
        "size": SIZE,
        "npz_filename": npz_path.name,
        "medmnist_version": "3.0.2",
        "expected_md5": "0" * 32,
        "file_sha256": "0" * 64,
        "file_bytes": npz_path.stat().st_size,
        "image_shape": [SIZE, SIZE],
        "split_sizes": counts,
        "class_names": [f"organ-{i}" for i in range(NUM_CLASSES)],
        "n_channels": 1,
        "task": "multi-class",
        "license": "CC BY 4.0",
        "source_url": "https://example.invalid/organamnist.npz",
        "split_note": "scan-level splits",
    }
    (tmp_path / "organamnist.manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_npz_filename_suffix_rule() -> None:
    """28 is the unsuffixed default; every larger variant is suffixed."""
    assert dm.npz_filename("organamnist", 28) == "organamnist.npz"
    assert dm.npz_filename("organamnist", 224) == "organamnist_224.npz"


def test_load_split_squeezes_labels(archive_root: Path) -> None:
    """MedMNIST's (N, 1) labels must arrive as a flat int64 vector.

    Left as (N, 1) they would broadcast against predictions in the loss and
    silently compute something other than what was intended.
    """
    split = dm.load_split("organamnist", SIZE, archive_root, "train")

    assert split.images.shape == (40, SIZE, SIZE)
    assert split.labels.shape == (40,)
    assert split.labels.dtype == np.int64
    assert len(split) == 40


def test_all_splits_load(archive_root: Path) -> None:
    expected = {"train": 40, "val": 12, "test": 20}
    for name, count in expected.items():
        assert len(dm.load_split("organamnist", SIZE, archive_root, name)) == count


def test_unknown_split_is_rejected(archive_root: Path) -> None:
    """Guards against a custom split slipping in.

    Resplitting would break comparability with the published baselines and
    would draw at slice level, putting the same patient on both sides.
    """
    with pytest.raises(ValueError, match="expected one of"):
        dm.load_split("organamnist", SIZE, archive_root, "holdout")


def test_missing_archive_message_contains_the_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="organ-service download"):
        dm.load_split("organamnist", SIZE, tmp_path, "train")


def test_missing_manifest_message_contains_the_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="organ-service download"):
        dm.load_manifest("organamnist", SIZE, tmp_path)


def test_manifest_round_trip(archive_root: Path) -> None:
    provenance = dm.load_manifest("organamnist", SIZE, archive_root)

    assert provenance.num_classes == NUM_CLASSES
    assert provenance.license == "CC BY 4.0"
    assert provenance.split_sizes["train"] == 40
    assert "file_sha256" in provenance.to_dict()


def test_manifest_ignores_unknown_keys(archive_root: Path) -> None:
    """Extra manifest fields must not break loading.

    The manifest is written by a script that may grow new fields; a strict
    reader would turn that into a crash on old checkouts.
    """
    path = archive_root / "organamnist.manifest.json"
    raw = json.loads(path.read_text())
    raw["some_future_field"] = 42
    path.write_text(json.dumps(raw))

    assert dm.load_manifest("organamnist", SIZE, archive_root).num_classes == NUM_CLASSES


def test_class_distribution_length(archive_root: Path) -> None:
    split = dm.load_split("organamnist", SIZE, archive_root, "train")
    counts = dm.class_distribution(split, NUM_CLASSES)

    assert len(counts) == NUM_CLASSES
    assert sum(counts) == len(split)


# --- Augmentation sampling -------------------------------------------------


def test_generator_is_reproducible_per_sample() -> None:
    """Same (seed, epoch, index) must yield the same transform.

    This is what makes augmented training independent of worker count: the
    draw is derived from the sample's coordinates, not from a shared stream
    whose position depends on iteration order.
    """
    config = augment.AugmentConfig()

    first = augment.sample_affine(config, augment.generator_for(42, 3, 17))
    second = augment.sample_affine(config, augment.generator_for(42, 3, 17))

    assert first == second


def test_different_epochs_give_different_transforms() -> None:
    config = augment.AugmentConfig()

    epoch_one = augment.sample_affine(config, augment.generator_for(42, 1, 17))
    epoch_two = augment.sample_affine(config, augment.generator_for(42, 2, 17))

    assert epoch_one != epoch_two


def test_samples_stay_within_configured_bounds() -> None:
    config = augment.AugmentConfig(rotation_deg=10.0, translate=0.1)

    for index in range(200):
        params = augment.sample_affine(config, augment.generator_for(0, 0, index))
        assert abs(params.rotation_deg) <= 10.0
        assert abs(params.translate_x) <= 0.1
        assert abs(params.translate_y) <= 0.1


def test_zero_config_returns_identity() -> None:
    """Disabling augmentation must produce the exact inference-time transform."""
    config = augment.AugmentConfig(rotation_deg=0.0, translate=0.0)
    assert augment.sample_affine(config, augment.generator_for(0, 0, 0)) is IDENTITY
