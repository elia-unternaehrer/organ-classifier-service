"""Export a trained checkpoint to ONNX.

    uv run --extra train python scripts/export_onnx.py \
        --checkpoint runs/resnet18_224/checkpoint.pt

Everything the exported artefact needs is read out of the checkpoint: the
resolution, the class names, the normalisation constants and the run name that
becomes the filename. Nothing is passed twice, so nothing can be passed
inconsistently.

The parity check runs as part of the export rather than only in the test
suite. An artefact that disagrees with the network it came from should never
reach disk in the first place, and finding out at export time costs seconds
where finding out after deployment costs a debugging session.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch

from organ_service import __version__
from organ_service.model_meta import SCHEMA_VERSION, ModelMetadata

OPSET = 17
PARITY_TOLERANCE = 1e-4
PARITY_SAMPLES = 8


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_model(checkpoint: dict) -> torch.nn.Module:
    """Rebuild the network described by a checkpoint and load its weights."""
    config = checkpoint["config"]
    model = timm.create_model(
        config["model_name"],
        pretrained=False,
        num_classes=checkpoint["num_classes"],
        in_chans=1,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def export(model: torch.nn.Module, image_size: int, destination: Path, opset: int = OPSET) -> str:
    """Write the graph with a dynamic batch axis.

    Batch size is left symbolic so the same artefact serves one request at a
    time in production and a whole calibration set during quantisation. Height
    and width stay fixed: they are part of the model's contract, and pinning
    them lets the runtime specialise its kernels.

    Torch defaults to its dynamo-based exporter from 2.6 onward, which needs
    ``onnxscript``. That path is selected explicitly rather than left to the
    default, so the behaviour does not shift under a torch upgrade, with the
    older TorchScript exporter kept as a fallback.

    Returns:
        Which exporter produced the file. Recorded next to the artefact,
        because the two can emit structurally different graphs.
    """
    dummy = torch.zeros(1, 1, image_size, image_size, dtype=torch.float32)
    common = {
        "input_names": ["input"],
        "output_names": ["logits"],
        "dynamic_axes": {"input": {0: "batch"}, "logits": {0: "batch"}},
        "opset_version": opset,
    }

    try:
        torch.onnx.export(model, dummy, str(destination), dynamo=True, **common)
        return "dynamo"
    except Exception as exc:
        print(f"dynamo exporter unavailable ({exc});\nfalling back to torchscript")
        torch.onnx.export(
            model,
            dummy,
            str(destination),
            dynamo=False,
            do_constant_folding=True,
            **common,
        )
        return "torchscript"


def attach_metadata(path: Path, metadata: ModelMetadata) -> None:
    """Embed the serving configuration into the artefact."""
    model = onnx.load(str(path))
    # Clear any existing entries so re-export does not accumulate duplicates.
    del model.metadata_props[:]
    for key, value in metadata.to_props().items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(path))


def check_parity(model: torch.nn.Module, path: Path, image_size: int, seed: int = 0) -> float:
    """Compare torch and ONNX Runtime on the same inputs.

    Random inputs rather than real images on purpose: noise exercises the
    numeric range far more aggressively than in-distribution data, so an
    operator that was translated incorrectly shows up here even when it would
    have stayed hidden on actual CT slices.

    Returns:
        The largest absolute logit difference observed.
    """
    rng = np.random.default_rng(seed)
    batch = rng.standard_normal((PARITY_SAMPLES, 1, image_size, image_size)).astype(np.float32)

    with torch.no_grad():
        expected = model(torch.from_numpy(batch)).numpy()

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"input": batch})[0]

    return float(np.abs(expected - actual).max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--model-version",
        default="dev",
        help="release tag this artefact will be published under",
    )
    parser.add_argument("--opset", type=int, default=OPSET)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=PARITY_TOLERANCE,
        help="maximum tolerated absolute logit difference",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    preprocessing = checkpoint["preprocessing"]

    run_name = config["run_name"]
    image_size = preprocessing["image_size"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    destination = args.out_dir / f"{run_name}_fp32.onnx"

    print(f"checkpoint  {args.checkpoint}")
    print(f"run         {run_name}, epoch {checkpoint['epoch']}")
    print(f"val bacc    {checkpoint['val_balanced_accuracy']:.4f}")
    print(f"input       1x1x{image_size}x{image_size}, opset {args.opset}")

    model = load_model(checkpoint)
    exporter = export(model, image_size, destination, args.opset)
    print(f"exporter    {exporter}")

    metadata = ModelMetadata(
        schema_version=SCHEMA_VERSION,
        model_version=args.model_version,
        run_name=run_name,
        precision="fp32",
        image_size=image_size,
        norm_mean=preprocessing["norm_mean"],
        norm_std=preprocessing["norm_std"],
        class_names=checkpoint["class_names"],
        git_sha=git_sha(),
        package_version=__version__,
    )
    attach_metadata(destination, metadata)

    deviation = check_parity(model, destination, image_size)
    print(f"parity      max abs logit difference {deviation:.3e}")

    if deviation > args.tolerance:
        destination.unlink()
        print(
            f"\nFAILED: deviation exceeds tolerance {args.tolerance:.1e}. "
            f"Artefact deleted rather than published.",
            file=sys.stderr,
        )
        return 1

    size_mb = destination.stat().st_size / 1e6
    print(f"wrote       {destination} ({size_mb:.1f} MB)")

    sidecar = destination.with_suffix(".metadata.json")
    sidecar.write_text(
        json.dumps(metadata.to_props() | {"parity_max_abs_diff": deviation}, indent=2) + "\n"
    )
    print(f"wrote       {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
