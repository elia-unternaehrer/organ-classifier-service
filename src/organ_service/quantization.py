"""Post-training quantisation and the comparisons that justify it.

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

import dataclasses
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from organ_service.metrics import agreement_rate, balanced_accuracy, predictions_from_logits
from organ_service.model_meta import ModelMetadata

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

    Symbolic shape inference needs ``sympy``, which is declared as a
    dependency but is easy to end up without in a trimmed environment. Rather
    than fail the whole run, this falls back to the shape-inference-free path
    and reports which one was used, because that choice can change the
    resulting graph and therefore belongs in the report.

    Returns:
        Whether symbolic shape inference ran.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process

    try:
        quant_pre_process(str(source), str(destination), skip_symbolic_shape=False)
        return True
    except ImportError:
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


def attach_metadata(path: Path, metadata: ModelMetadata) -> None:
    """Re-embed serving metadata after quantisation.

    The quantisation tools rewrite the graph and drop ``metadata_props``, so
    without this step a quantised artefact would be unservable: the service
    refuses models whose preprocessing parameters it cannot read.
    """
    import onnx

    model = onnx.load(str(path))
    del model.metadata_props[:]
    for key, value in metadata.to_props().items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(path))


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
