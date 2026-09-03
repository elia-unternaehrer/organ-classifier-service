"""Export a trained checkpoint to ONNX.

    organ-service export --checkpoint runs/resnet18_224/checkpoint.pt

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
import onnxruntime as ort
import timm
import torch

from organ_service import __version__
from organ_service.model_meta import SCHEMA_VERSION, ModelMetadata, attach_metadata

OPSET = 18
"""Torch's dynamo exporter implements 18. Requesting 17 makes it export at 18
and then attempt a downgrade that fails on this graph, leaving the model at 18
anyway after a lot of noise. Asking for what the exporter produces is simpler
and avoids a conversion that buys nothing: ONNX Runtime has supported 18 since
1.14."""
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


def export(
    model: torch.nn.Module,
    image_size: int,
    destination: Path,
    opset: int = OPSET,
    exporter: str = "torchscript",
) -> str:
    """Write the graph with a dynamic batch axis.

    Batch size is left symbolic so the same artefact serves one request at a
    time in production and a whole calibration set during quantisation. Height
    and width stay fixed: they are part of the model's contract, and pinning
    them lets the runtime specialise its kernels.

    TorchScript is the default despite being the older path, and the reason is
    downstream rather than here. The dynamo exporter produces a graph that runs
    correctly, with parity against the torch model at around 2e-06, but one
    that ONNX's own shape inference rejects: it reports a mismatch at the
    classifier head, ``(512) vs (11)``. Quantisation calls that inference
    internally, so a dynamo-exported artefact cannot be quantised, and
    quantisation is not optional in this pipeline.

    So the choice is between a modern exporter whose output is a dead end and
    a deprecated one whose output completes the pipeline. The dynamo path stays
    available behind ``--exporter dynamo`` so the decision is recorded rather
    than hidden, and so it can be retried against a future torch release.

    Parity against the torch model is checked immediately after export either
    way, and that is the actual correctness guarantee; which exporter produced
    the graph is provenance.

    Returns:
        Which exporter produced the file, recorded next to the artefact
        because the two can emit structurally different graphs.
    """
    dummy = torch.zeros(1, 1, image_size, image_size, dtype=torch.float32)
    common = {
        "input_names": ["input"],
        "output_names": ["logits"],
        "opset_version": opset,
    }

    if exporter == "dynamo":
        torch.onnx.export(
            model,
            dummy,
            str(destination),
            dynamo=True,
            dynamic_shapes=({0: torch.export.Dim.AUTO},),
            **common,
        )
        return "dynamo"

    torch.onnx.export(
        model,
        dummy,
        str(destination),
        dynamo=False,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        **common,
    )
    return "torchscript"


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


def main(argv: list[str] | None = None) -> int:
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
        "--exporter",
        choices=("torchscript", "dynamo"),
        default="torchscript",
        help="ONNX exporter; dynamo output currently cannot be quantised",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=PARITY_TOLERANCE,
        help="maximum tolerated absolute logit difference",
    )
    args = parser.parse_args(argv)

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
    exporter = export(model, image_size, destination, args.opset, args.exporter)
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
