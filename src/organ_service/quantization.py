"""Post-training quantisation and the comparisons that justify it.

    organ-service quantize --model artifacts/resnet18_224_fp32.onnx

Free of torch: quantisation operates on the exported ONNX graph, and
calibration needs nothing but the raw arrays and the shared transform. The
artefact pipeline therefore does not depend on the training framework, which
is what lets it run in CI.

Two schemes are produced.

**Dynamic** quantises weights ahead of time and derives activation scales at
runtime. No calibration data, no risk of a calibration set that misrepresents
production traffic, but the per-batch scale computation costs time on every
request.

**Static** quantises activations too, using ranges estimated from a
calibration pass. That is where the CPU speedup actually comes from, at the
price of a calibration set and a real, if usually small, accuracy cost.

Comparing them needs a different test from the fp32 parity check. Quantised
outputs *must* differ numerically, so asserting closeness of logits is the
wrong question. What matters is whether the decision changes, which is what
``metrics.agreement_rate`` measures.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from organ_service.data import load_split
from organ_service.metrics import agreement_rate, balanced_accuracy, predictions_from_logits
from organ_service.model_meta import ModelMetadata, attach_metadata

CALIBRATION_BATCH = 8
EVAL_BATCH = 32


@dataclass(frozen=True)
class EvaluationResult:
    """How one artefact behaves relative to the fp32 baseline."""

    precision: str
    size_bytes: int
    balanced_accuracy: float
    agreement_with_baseline: float
    max_abs_logit_diff: float

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1e6


class ArrayCalibrationReader:
    """Feeds preprocessed batches to the static quantiser.

    Calibration samples must come from the training split. Estimating
    activation ranges is fitting parameters to data, so drawing them from
    validation would contaminate the split that selection relies on, and
    drawing them from test would be plainly wrong.

    Implements ``onnxruntime.quantization.CalibrationDataReader`` structurally
    rather than by inheritance, so that importing this module does not pull in
    the quantisation subpackage.
    """

    def __init__(self, batches: list[dict[str, np.ndarray]]) -> None:
        self._batches = batches
        self._iterator: Iterator[dict[str, np.ndarray]] = iter(batches)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iterator, None)

    def rewind(self) -> None:
        self._iterator = iter(self._batches)


def build_calibration_batches(
    images: np.ndarray,
    image_size: int,
    count: int,
    seed: int,
    batch_size: int = CALIBRATION_BATCH,
) -> list[dict[str, np.ndarray]]:
    """Draw and preprocess a calibration set.

    Args:
        images: Raw uint8 training images, ``(N, H, W)``.
        count: How many samples to calibrate on. A few hundred is the usual
            recommendation; the estimate is of activation ranges, not of
            anything that needs statistical power.
        seed: Recorded in the run's metrics so the calibration set can be
            reconstructed exactly.
    """
    from organ_service.preprocessing import preprocess_from_array

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(images), size=min(count, len(images)), replace=False)

    batches = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        stacked = np.stack(
            [preprocess_from_array(np.asarray(images[i]), size=image_size) for i in chunk]
        )
        batches.append({"input": stacked.astype(np.float32)})
    return batches


def preprocess_graph(source: Path, destination: Path) -> bool:
    """Run ONNX Runtime's pre-quantisation graph cleanup.

    Symbolic shape inference and constant folding. Skipping it produces the
    "consider pre-processing before quantization" warning, and on
    convolutional graphs it affects which nodes end up quantised at all.

    Symbolic shape inference is best-effort here. It needs ``sympy``, and it
    does not always succeed on dynamo-exported graphs. Rather than fail the
    whole run this falls back to the inference-free path and reports which one
    was used, because the choice can change the resulting graph and therefore
    belongs in the report.

    Returns:
        Whether symbolic shape inference ran.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process

    try:
        quant_pre_process(str(source), str(destination), skip_symbolic_shape=False)
        return True
    except Exception as exc:
        # Two known causes, both non-fatal. sympy may be absent, and symbolic
        # inference raises "Incomplete symbolic shape inference" on graphs the
        # dynamo exporter emits. Neither prevents quantisation: the pass is an
        # optimisation, and the shapes it would have inferred are already
        # static everywhere except the batch axis.
        print(f"symbolic shape inference unavailable ({exc}); continuing without it")
        quant_pre_process(str(source), str(destination), skip_symbolic_shape=True)
        return False


def quantise_dynamic(source: Path, destination: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(source), str(destination), weight_type=QuantType.QInt8)


def quantise_static(source: Path, destination: Path, reader: ArrayCalibrationReader) -> None:
    """Quantise weights and activations using calibrated ranges.

    Per-channel weight scales are enabled: convolution kernels in a trained
    network have per-filter magnitudes that differ by orders of magnitude, and
    a single tensor-wide scale wastes most of the int8 range on the largest
    filter. The QDQ format is used because it is what the CPU execution
    provider optimises for.
    """
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    quantize_static(
        str(source),
        str(destination),
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )


def infer_logits(
    path: Path, images: np.ndarray, image_size: int, batch_size: int = EVAL_BATCH
) -> np.ndarray:
    """Run a whole split through an artefact, batched."""
    from organ_service.preprocessing import preprocess_from_array

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        stacked = np.stack(
            [preprocess_from_array(np.asarray(img), size=image_size) for img in chunk]
        ).astype(np.float32)
        outputs.append(session.run(["logits"], {"input": stacked})[0])
    return np.concatenate(outputs)


def evaluate(
    path: Path,
    images: np.ndarray,
    labels: np.ndarray,
    image_size: int,
    precision: str,
    baseline_logits: np.ndarray | None = None,
) -> tuple[EvaluationResult, np.ndarray]:
    """Score one artefact, optionally against the fp32 baseline."""
    logits = infer_logits(path, images, image_size)
    reference = baseline_logits if baseline_logits is not None else logits

    return (
        EvaluationResult(
            precision=precision,
            size_bytes=path.stat().st_size,
            balanced_accuracy=balanced_accuracy(labels, predictions_from_logits(logits)),
            agreement_with_baseline=agreement_rate(reference, logits),
            max_abs_logit_diff=float(np.abs(reference - logits).max()),
        ),
        logits,
    )


def quantised_metadata(base: ModelMetadata, precision: str) -> ModelMetadata:
    return dataclasses.replace(base, precision=precision)


@contextmanager
def temporary_graph(suffix: str = ".onnx") -> Iterator[Path]:
    """A scratch path for the pre-processed graph, removed on exit.

    The cleaned graph is an intermediate that both quantisers read and nobody
    should ship, so its lifetime is scoped rather than left to a manual
    unlink that an early return could skip.
    """
    descriptor, name = tempfile.mkstemp(suffix=suffix)
    os.close(descriptor)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


# --- CLI ---------------------------------------------------------------
# Kept in the same module as the library it drives: the command is a thin
# wrapper, and splitting it out would only add a file whose sole content is
# argument parsing.


def read_metadata(path: Path) -> ModelMetadata:
    """Read serving metadata straight off an artefact.

    Goes through onnxruntime rather than the onnx package, the same way the
    service will, so a model that this command can read is one the service can
    load.
    """
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return ModelMetadata.from_props(session.get_modelmeta().custom_metadata_map)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="the fp32 artefact")
    parser.add_argument("--dataset", default="organamnist")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=512,
        help="training images used to estimate activation ranges",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=42,
        help="recorded in the report so the calibration set is reconstructible",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=0,
        help="validation images to compare on; 0 means the whole split",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="seed for subsampling the evaluation split",
    )
    args = parser.parse_args(argv)

    metadata = read_metadata(args.model)
    image_size = metadata.image_size
    run_name = metadata.run_name
    out_dir = args.model.parent

    print(f"model       {args.model} ({args.model.stat().st_size / 1e6:.2f} MB)")
    print(f"metadata    {metadata.summary()}")

    train = load_split(args.dataset, image_size, args.data_root, "train")
    val = load_split(args.dataset, image_size, args.data_root, "val")

    if args.eval_samples and args.eval_samples < len(val):
        # Sampled rather than sliced. MedMNIST orders its splits by CT scan, so
        # the first N images are the first two or three patients: a correlated
        # sample that says little about the split as a whole.
        rng = np.random.default_rng(args.eval_seed)
        picked = np.sort(rng.choice(len(val), size=args.eval_samples, replace=False))
        val_images, val_labels = val.images[picked], val.labels[picked]

        covered = len(np.unique(val_labels))
        expected = metadata.num_classes
        if covered < expected:
            print(
                f"warning: subset covers {covered} of {expected} classes; "
                f"balanced accuracy is averaged over the classes present and "
                f"is not comparable to a full-split number"
            )
    else:
        val_images, val_labels = val.images, val.labels
    print(f"calibrate   {args.calibration_samples} train images, seed {args.calibration_seed}")
    print(f"evaluate    {len(val_images)} validation images\n")

    dynamic_path = out_dir / f"{run_name}_int8_dynamic.onnx"
    static_path = out_dir / f"{run_name}_int8_static.onnx"

    with temporary_graph() as prepared:
        symbolic = preprocess_graph(args.model, prepared)
        if not symbolic:
            print("note: proceeding without symbolic shape inference")

        print("quantising dynamic ...")
        quantise_dynamic(prepared, dynamic_path)
        attach_metadata(dynamic_path, quantised_metadata(metadata, "int8_dynamic"))

        print("quantising static ...")
        reader = ArrayCalibrationReader(
            build_calibration_batches(
                train.images, image_size, args.calibration_samples, args.calibration_seed
            )
        )
        quantise_static(prepared, static_path, reader)
        attach_metadata(static_path, quantised_metadata(metadata, "int8_static"))

    print("evaluating ...\n")
    baseline, baseline_logits = evaluate(args.model, val_images, val_labels, image_size, "fp32")
    results = [baseline]
    for path, precision in ((dynamic_path, "int8_dynamic"), (static_path, "int8_static")):
        result, _ = evaluate(path, val_images, val_labels, image_size, precision, baseline_logits)
        results.append(result)

    header = f"{'precision':<14}{'size MB':>9}{'val bacc':>10}{'agree':>9}{'max Δlogit':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.precision:<14}{r.size_mb:>9.2f}{r.balanced_accuracy:>10.4f}"
            f"{r.agreement_with_baseline:>9.4f}{r.max_abs_logit_diff:>12.3e}"
        )

    report = {
        "run_name": run_name,
        "image_size": image_size,
        "calibration": {
            "split": "train",
            "samples": args.calibration_samples,
            "seed": args.calibration_seed,
        },
        "evaluation": {
            "split": "val",
            "samples": len(val_images),
            "subsampled": bool(args.eval_samples and args.eval_samples < len(val)),
            "seed": args.eval_seed,
        },
        "symbolic_shape_inference": symbolic,
        "results": [
            {
                "precision": r.precision,
                "size_bytes": r.size_bytes,
                "size_mb": round(r.size_mb, 3),
                "balanced_accuracy": r.balanced_accuracy,
                "agreement_with_fp32": r.agreement_with_baseline,
                "max_abs_logit_diff": r.max_abs_logit_diff,
            }
            for r in results
        ],
    }
    report_path = out_dir / f"{run_name}_quantization.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nwrote       {dynamic_path}")
    print(f"wrote       {static_path}")
    print(f"wrote       {report_path}")

    baseline_bacc = baseline.balanced_accuracy
    for r in results[1:]:
        delta = (r.balanced_accuracy - baseline_bacc) * 100
        print(
            f"{r.precision:<14}{r.size_mb / baseline.size_mb:.2f}x size, "
            f"{delta:+.2f} pp balanced accuracy"
        )

    # Deliberately no threshold assertion here. What counts as an acceptable
    # accuracy cost is a deployment decision made against measured latency,
    # not something this script can decide on its own.
    return 0
