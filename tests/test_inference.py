"""Tests for the inference layer.

Built on small hand-made ONNX graphs so the whole registry, including the
multi-model paths, runs in the default CI job without torch and without the
real 45 MB artefacts.
"""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from organ_service.model_meta import SCHEMA_VERSION, ModelMetadata
from organ_service.serve.inference import (
    LoadedModel,
    ModelRegistry,
    RuntimeConfig,
    softmax,
)

NUM_CLASSES = 4
CLASS_NAMES = ["liver", "spleen", "kidney-left", "kidney-right"]


def build_model(
    path: Path,
    run_name: str,
    precision: str,
    image_size: int,
    class_names: list[str] | None = None,
) -> Path:
    """A graph that maps any input to fixed logits, with metadata attached.

    The arithmetic is irrelevant; what the tests exercise is metadata-driven
    configuration, resolution handling and registry behaviour.
    """
    names = class_names or CLASS_NAMES
    rng = np.random.default_rng(abs(hash(run_name + precision)) % 1000)

    weight = rng.standard_normal((1, len(names))).astype(np.float32)
    # From opset 18, ReduceMean takes its axes as an input rather than an
    # attribute.
    axes = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes")

    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["input", "axes"], ["pooled"], keepdims=0),
            helper.make_node("MatMul", ["pooled", "w"], ["logits"]),
        ],
        "stub",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["batch", 1, image_size, image_size]
            )
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(names)])],
        initializer=[numpy_helper.from_array(weight, "w"), axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)

    metadata = ModelMetadata(
        schema_version=SCHEMA_VERSION,
        model_version="v0.1.0",
        run_name=run_name,
        precision=precision,
        image_size=image_size,
        norm_mean=0.5,
        norm_std=0.5,
        class_names=names,
        git_sha="abc123",
        package_version="0.1.0",
    )
    for key, value in metadata.to_props().items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value

    onnx.save(model, str(path))
    return path


@pytest.fixture
def fp32_path(tmp_path: Path) -> Path:
    return build_model(tmp_path / "fp32.onnx", "resnet18_224", "fp32", 224)


@pytest.fixture
def int8_path(tmp_path: Path) -> Path:
    return build_model(tmp_path / "int8.onnx", "resnet18_224", "int8_static", 224)


@pytest.fixture
def small_path(tmp_path: Path) -> Path:
    return build_model(tmp_path / "small.onnx", "resnet18_64", "fp32", 64)


@pytest.fixture
def image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (28, 28), dtype=np.uint8), mode="L")


# --- softmax ---------------------------------------------------------------


def test_softmax_sums_to_one() -> None:
    assert softmax(np.array([1.0, 2.0, 3.0])).sum() == pytest.approx(1.0)


def test_softmax_survives_large_logits() -> None:
    """Stability matters: quantised models produce wider logit ranges.

    The naive formulation overflows to nan here, which would surface as a
    500 on exactly the models the demo is meant to show off.
    """
    result = softmax(np.array([1000.0, 1001.0, 999.0]))
    assert np.isfinite(result).all()
    assert result.sum() == pytest.approx(1.0)


def test_softmax_preserves_ordering() -> None:
    logits = np.array([0.5, 3.0, -1.0])
    assert softmax(logits).argmax() == logits.argmax()


# --- LoadedModel -----------------------------------------------------------


def test_model_configures_itself_from_metadata(fp32_path: Path) -> None:
    model = LoadedModel(fp32_path)
    assert model.metadata.image_size == 224
    assert model.model_id == "resnet18_224_fp32"


def test_prediction_covers_every_class(fp32_path: Path, image: Image.Image) -> None:
    """All classes are returned, not just the top one.

    Truncating here would make the response useless for anything except this
    demo's bar chart.
    """
    result = LoadedModel(fp32_path).predict(image)

    assert len(result.predictions) == NUM_CLASSES
    assert {p.class_name for p in result.predictions} == set(CLASS_NAMES)
    assert sum(p.probability for p in result.predictions) == pytest.approx(1.0)


def test_predictions_are_sorted(fp32_path: Path, image: Image.Image) -> None:
    probabilities = [p.probability for p in LoadedModel(fp32_path).predict(image).predictions]
    assert probabilities == sorted(probabilities, reverse=True)


def test_timings_are_reported_separately(fp32_path: Path, image: Image.Image) -> None:
    """The split is the point: the two scale differently with resolution."""
    result = LoadedModel(fp32_path).predict(image)
    assert result.preprocess_ms > 0
    assert result.inference_ms > 0


def test_input_size_is_taken_from_the_artefact(small_path: Path, image: Image.Image) -> None:
    """A 64px model must resize to 64 without being told.

    This is what makes artefacts interchangeable. If the size came from
    service configuration instead, a mismatch would produce no error, just
    silently wrong predictions.
    """
    assert LoadedModel(small_path).predict(image).predictions


def test_arbitrary_upload_sizes_are_accepted(fp32_path: Path) -> None:
    """Browser uploads are whatever the visitor happened to have."""
    model = LoadedModel(fp32_path)
    for size in ((28, 28), (512, 384), (7, 7)):
        assert model.predict(Image.new("RGB", size, color=128)).predictions


# --- ModelRegistry ---------------------------------------------------------


def test_registry_exposes_every_model(fp32_path: Path, int8_path: Path) -> None:
    registry = ModelRegistry.from_paths([fp32_path, int8_path])
    assert registry.ids == ["resnet18_224_fp32", "resnet18_224_int8_static"]


def test_first_model_is_default(fp32_path: Path, int8_path: Path) -> None:
    registry = ModelRegistry.from_paths([fp32_path, int8_path])
    assert registry.default_id == "resnet18_224_fp32"
    assert registry.get().model_id == "resnet18_224_fp32"


def test_explicit_default(fp32_path: Path, int8_path: Path) -> None:
    registry = ModelRegistry.from_paths(
        [fp32_path, int8_path], default_id="resnet18_224_int8_static"
    )
    assert registry.get().model_id == "resnet18_224_int8_static"


def test_unknown_default_is_refused(fp32_path: Path) -> None:
    with pytest.raises(ValueError, match="not among"):
        ModelRegistry.from_paths([fp32_path], default_id="nope")


def test_unknown_model_lists_the_alternatives(fp32_path: Path, int8_path: Path) -> None:
    """An API client that guessed wrong needs to know what it may ask for."""
    registry = ModelRegistry.from_paths([fp32_path, int8_path])
    with pytest.raises(KeyError, match="available"):
        registry.get("resnet18_224_int8_dynamic")


def test_models_at_different_resolutions_coexist(
    fp32_path: Path, small_path: Path, image: Image.Image
) -> None:
    """The registry holds resolution variants as readily as precision ones."""
    registry = ModelRegistry.from_paths([fp32_path, small_path])

    assert registry.get("resnet18_224_fp32").metadata.image_size == 224
    assert registry.get("resnet18_64_fp32").metadata.image_size == 64
    assert registry.get("resnet18_64_fp32").predict(image).predictions


def test_mismatched_class_names_are_refused(tmp_path: Path, fp32_path: Path) -> None:
    """A dropdown switching between differently labelled models is nonsense.

    Worse, the mismatch would be invisible until someone compared two
    predictions and found the labels disagreed.
    """
    other = build_model(
        tmp_path / "other.onnx",
        "pathmnist_224",
        "fp32",
        224,
        class_names=["a", "b", "c", "d"],
    )
    with pytest.raises(ValueError, match="different class names"):
        ModelRegistry.from_paths([fp32_path, other])


def test_empty_registry_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ModelRegistry([])


def test_missing_file_is_reported_with_its_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"absent\.onnx"):
        ModelRegistry.from_paths([tmp_path / "absent.onnx"])


def test_describe_marks_the_default(fp32_path: Path, int8_path: Path) -> None:
    registry = ModelRegistry.from_paths([fp32_path, int8_path])
    described = registry.describe()

    assert [entry["is_default"] for entry in described] == [True, False]
    assert described[0]["size_bytes"] > 0
    assert described[1]["precision"] == "int8_static"


def test_runtime_config_reaches_the_session(fp32_path: Path) -> None:
    """Thread and arena settings are what keep the service inside its quota."""
    options = RuntimeConfig(intra_op_threads=2, enable_memory_arena=True)
    assert options.to_session_options().intra_op_num_threads == 2
    assert LoadedModel(fp32_path, options).predict(Image.new("L", (32, 32))).predictions
