"""Tests for the quantisation step.

A small convolutional graph is built and genuinely quantised here, both ways.
That is slower than mocking but worth it: the failure modes these tests exist
to catch, metadata being dropped by the quantiser and the calibration reader
being exhausted early, only appear when the real tooling runs.

No torch is involved, so these stay in the default CI job.
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from organ_service import metrics
from organ_service import quantization as q
from organ_service.model_meta import SCHEMA_VERSION, ModelMetadata

IMAGE_SIZE = 32
NUM_CLASSES = 4


@pytest.fixture
def metadata() -> ModelMetadata:
    return ModelMetadata(
        schema_version=SCHEMA_VERSION,
        model_version="v0.1.0",
        run_name="tinynet_32",
        precision="fp32",
        image_size=IMAGE_SIZE,
        norm_mean=0.5,
        norm_std=0.5,
        class_names=[f"organ-{i}" for i in range(NUM_CLASSES)],
        git_sha="abc123",
        package_version="0.1.0",
    )


@pytest.fixture
def fp32_model(tmp_path: Path, metadata: ModelMetadata) -> Path:
    """A convolutional graph, so the quantisers exercise Conv rather than only
    MatMul. A purely linear fixture would not represent the real model."""
    rng = np.random.default_rng(0)
    nodes, initialisers = [], []
    channels_in, previous = 1, "input"

    for index, channels_out in enumerate([8, 16]):
        weight = rng.standard_normal((channels_out, channels_in, 3, 3)).astype(np.float32)
        initialisers.append(numpy_helper.from_array(weight, f"w{index}"))
        nodes.append(
            helper.make_node("Conv", [previous, f"w{index}"], [f"c{index}"], pads=[1, 1, 1, 1])
        )
        nodes.append(helper.make_node("Relu", [f"c{index}"], [f"r{index}"]))
        previous, channels_in = f"r{index}", channels_out

    nodes.append(helper.make_node("GlobalAveragePool", [previous], ["pooled"]))
    nodes.append(helper.make_node("Flatten", ["pooled"], ["flat"], axis=1))
    initialisers.append(
        numpy_helper.from_array(rng.standard_normal((16, NUM_CLASSES)).astype(np.float32), "head")
    )
    nodes.append(helper.make_node("MatMul", ["flat", "head"], ["logits"]))

    graph = helper.make_graph(
        nodes,
        "tinynet",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["batch", 1, IMAGE_SIZE, IMAGE_SIZE]
            )
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", NUM_CLASSES])],
        initializer=initialisers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    for key, value in metadata.to_props().items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value

    path = tmp_path / "fp32.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def images() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (24, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)


@pytest.fixture
def labels() -> np.ndarray:
    rng = np.random.default_rng(2)
    return rng.integers(0, NUM_CLASSES, 24).astype(np.int64)


def test_calibration_reader_is_exhaustible(images: np.ndarray) -> None:
    """The quantiser stops on None; a reader that never ends would hang."""
    reader = q.ArrayCalibrationReader(
        q.build_calibration_batches(images, IMAGE_SIZE, count=16, seed=0, batch_size=4)
    )

    seen = 0
    while (batch := reader.get_next()) is not None:
        assert batch["input"].dtype == np.float32
        assert batch["input"].shape[1:] == (1, IMAGE_SIZE, IMAGE_SIZE)
        seen += batch["input"].shape[0]

    assert seen == 16


def test_calibration_is_reproducible(images: np.ndarray) -> None:
    """Same seed, same calibration set.

    The seed is recorded in the report, which only means something if it
    actually determines the draw.
    """
    first = q.build_calibration_batches(images, IMAGE_SIZE, 16, seed=7, batch_size=4)
    second = q.build_calibration_batches(images, IMAGE_SIZE, 16, seed=7, batch_size=4)

    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a["input"], b["input"])


def test_dynamic_quantisation_shrinks_and_keeps_metadata(
    fp32_model: Path, tmp_path: Path, metadata: ModelMetadata
) -> None:
    destination = tmp_path / "dynamic.onnx"
    q.quantise_dynamic(fp32_model, destination)
    q.attach_metadata(destination, q.quantised_metadata(metadata, "int8_dynamic"))

    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    recovered = ModelMetadata.from_props(session.get_modelmeta().custom_metadata_map)

    assert recovered.precision == "int8_dynamic"
    assert recovered.image_size == IMAGE_SIZE
    assert recovered.class_names == metadata.class_names


def test_static_quantisation_keeps_metadata(
    fp32_model: Path, tmp_path: Path, images: np.ndarray, metadata: ModelMetadata
) -> None:
    """The guard that matters most here.

    The quantisation tools rewrite the graph and drop metadata_props. Without
    re-attachment the artefact loads but the service refuses it, and the cause
    is far from obvious at that point.
    """
    destination = tmp_path / "static.onnx"
    reader = q.ArrayCalibrationReader(
        q.build_calibration_batches(images, IMAGE_SIZE, 16, seed=0, batch_size=4)
    )

    q.quantise_static(fp32_model, destination, reader)
    q.attach_metadata(destination, q.quantised_metadata(metadata, "int8_static"))

    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    recovered = ModelMetadata.from_props(session.get_modelmeta().custom_metadata_map)

    assert recovered.precision == "int8_static"
    assert recovered.norm_mean == metadata.norm_mean


def test_quantised_model_still_runs(fp32_model: Path, tmp_path: Path, images: np.ndarray) -> None:
    destination = tmp_path / "dynamic.onnx"
    q.quantise_dynamic(fp32_model, destination)

    logits = q.infer_logits(destination, images, IMAGE_SIZE, batch_size=8)
    assert logits.shape == (len(images), NUM_CLASSES)


def test_agreement_rate_bounds() -> None:
    """Identical predictions agree fully; inverted ones not at all."""
    baseline = np.array([[3.0, 1.0], [0.5, 2.0], [1.0, 0.0]])

    assert metrics.agreement_rate(baseline, baseline) == 1.0
    assert metrics.agreement_rate(baseline, baseline[:, ::-1]) == 0.0


def test_agreement_ignores_logit_magnitude() -> None:
    """Scaling every logit changes the numbers but no decision.

    Exactly why agreement replaces a closeness assertion for quantised models:
    the outputs are meant to move, only the argmax must hold.
    """
    baseline = np.array([[3.0, 1.0], [0.5, 2.0]])
    assert metrics.agreement_rate(baseline, baseline * 7.0 + 100.0) == 1.0


def test_balanced_accuracy_argument_order(labels: np.ndarray) -> None:
    """Guards the (y_true, y_pred) order.

    Balanced accuracy is not symmetric in its arguments, and a transposition
    would corrupt checkpoint selection, the quantisation comparison and the
    model card at once while still producing plausible-looking numbers.
    """
    predictions = labels.copy()
    predictions[:4] = (predictions[:4] + 1) % NUM_CLASSES

    forward = metrics.balanced_accuracy(labels, predictions)
    backward = metrics.balanced_accuracy(predictions, labels)

    assert forward == pytest.approx(
        __import__("sklearn.metrics", fromlist=["x"]).balanced_accuracy_score(labels, predictions)
    )
    assert forward != backward


def test_evaluate_reports_baseline_agreement(
    fp32_model: Path, images: np.ndarray, labels: np.ndarray
) -> None:
    """Evaluating the baseline against itself is the degenerate case."""
    result, logits = q.evaluate(fp32_model, images, labels, IMAGE_SIZE, "fp32")

    assert result.agreement_with_baseline == 1.0
    assert result.max_abs_logit_diff == 0.0
    assert logits.shape == (len(images), NUM_CLASSES)
    assert 0.0 <= result.balanced_accuracy <= 1.0
