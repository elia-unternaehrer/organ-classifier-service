"""Download a MedMNIST dataset and record its provenance.

Run once per resolution before training:

    uv run --extra train python scripts/download_data.py --size 224

The download itself is delegated to the ``medmnist`` package, which knows the
Zenodo URLs and verifies the published MD5. What this script adds is the
manifest: a JSON sidecar recording exactly which artefact ended up on disk.

That manifest is the reason this project does not use DVC. The dataset is an
immutable public artefact pinned by package version and checksum, so there is
no evolving data to version. What actually needs answering is "which bytes
produced these numbers", and a recorded hash answers it without asking anyone
to configure a DVC remote in order to read the repository.

The manifest also decouples the rest of the codebase from ``medmnist``.
Importing that package pulls in torch, so having ``data.py`` read class names
and licensing from JSON instead keeps the data layer torch-free and testable
in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

AVAILABLE_SIZES = (28, 64, 128, 224)
DEFAULT_DATASET = "organamnist"
SPLITS = ("train", "val", "test")


def npz_filename(dataset: str, size: int) -> str:
    """Reproduce MedMNIST's on-disk naming.

    The 28-pixel variant carries no size suffix, every larger one does.
    """
    suffix = "" if size == 28 else f"_{size}"
    return f"{dataset}{suffix}.npz"


def file_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Hash a file in chunks.

    The 224-pixel variant is several gigabytes, so it is never read whole.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(dataset: str, size: int, root: Path) -> dict:
    """Collect everything needed to identify this artefact later."""
    import medmnist
    import numpy as np
    from medmnist import INFO

    info = INFO[dataset]
    path = root / npz_filename(dataset, size)

    md5_key = "MD5" if size == 28 else f"MD5_{size}"
    url_key = "url" if size == 28 else f"url_{size}"

    with np.load(path) as archive:
        split_sizes = {split: int(archive[f"{split}_images"].shape[0]) for split in SPLITS}
        image_shape = list(archive["train_images"].shape[1:])

    return {
        "dataset": dataset,
        "size": size,
        "npz_filename": path.name,
        "medmnist_version": medmnist.__version__,
        "expected_md5": info[md5_key],
        "file_sha256": file_sha256(path),
        "file_bytes": path.stat().st_size,
        "image_shape": image_shape,
        "split_sizes": split_sizes,
        "class_names": [info["label"][str(i)] for i in range(len(info["label"]))],
        "n_channels": info["n_channels"],
        "task": info["task"],
        "license": info["license"],
        "source_url": info[url_key],
        # Splits are drawn at CT-scan level rather than slice level, so no
        # patient appears in more than one split. Recording this matters:
        # slice-level splitting is the most common way medical imaging results
        # end up inflated, and a reader cannot tell from the numbers alone.
        "split_note": (
            "Official MedMNIST splits, drawn at CT-scan level "
            "(115 train / 16 val / 70 test scans). No patient overlap."
        ),
    }


def download(dataset: str, size: int, root: Path, retries: int = 3) -> Path:
    """Fetch the archive unless it is already present.

    ``medmnist`` verifies the published MD5 on download and re-fetches on
    mismatch, so a corrupted partial download does not silently persist.

    Zenodo returns 504 on the multi-gigabyte variants often enough that a bare
    call is unreliable, hence the retries. Its downloader cannot resume, so a
    timeout at 90 percent costs the whole transfer; when retries run out the
    fallback printed below uses curl, which can.
    """
    from medmnist import INFO

    root.mkdir(parents=True, exist_ok=True)
    path = root / npz_filename(dataset, size)

    if path.exists():
        print(f"already present: {path}")
        return path

    info = INFO[dataset]
    python_class = info["python_class"]
    module = __import__("medmnist", fromlist=[python_class])
    dataset_class = getattr(module, python_class)

    url_key = "url" if size == 28 else f"url_{size}"
    md5_key = "MD5" if size == 28 else f"MD5_{size}"

    for attempt in range(1, retries + 1):
        print(f"downloading {dataset} at size {size} into {root} (attempt {attempt}/{retries}) ...")
        try:
            # Instantiating any split triggers the download of the whole archive.
            dataset_class(split="train", download=True, size=size, root=str(root))
            return path
        except RuntimeError as exc:
            if attempt == retries:
                raise SystemExit(
                    f"\ndownload failed after {retries} attempts: {exc}\n\n"
                    f"Zenodo times out on the larger archives. Fetch it with a "
                    f"resumable client instead, then re-run this script to write "
                    f"the manifest:\n\n"
                    f'  curl -L -C - -o {path} "{info[url_key]}"\n'
                    f"  md5sum {path}   # expect {info[md5_key]}\n"
                ) from exc
            wait = 5 * attempt
            print(f"  failed, retrying in {wait}s ...")
            time.sleep(wait)

    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--size", type=int, default=224, choices=AVAILABLE_SIZES)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--force-manifest",
        action="store_true",
        help="rewrite the manifest even if it exists (rehashes the archive)",
    )
    args = parser.parse_args()

    path = download(args.dataset, args.size, args.root)
    manifest_path = path.with_suffix(".manifest.json")

    if manifest_path.exists() and not args.force_manifest:
        print(f"manifest already present: {manifest_path}")
        return 0

    print("hashing archive ...")
    manifest = build_manifest(args.dataset, args.size, args.root)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {manifest_path}")
    print(f"  sha256      {manifest['file_sha256']}")
    print(f"  splits      {manifest['split_sizes']}")
    print(f"  classes     {len(manifest['class_names'])}")
    print(f"  licence     {manifest['license']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
