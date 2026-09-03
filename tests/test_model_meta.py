"""Tests for the self-describing model artefact.

A tiny hand-built ONNX graph stands in for the real network. That keeps these
tests in the default CI job, which installs no torch, while still exercising
the path that matters: metadata written with the ``onnx`` package and read back
through ``onnxruntime``, exactly as the service will read it.
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from organ_service.model_meta import KEY_PREFIX, SCHEMA_VERSION, ModelMetadata

IMAGE_SIZE = 32
NUM_CLASSES = 4


@pytest.fixture
def metadata() -> ModelMetadata:
    return ModelMetadata(
        schema_version=SCHEMA_VERSION,
        model_version="v0.1.0",
        run_name="resnet18_224",
        precision="fp32",
        image_size=224,
        norm_mean=0.5,
        norm_std=0.5,
        class_names=[f"organ-{i}" for i in range(NUM_CLASSES)],
        git_sha="abc123",
        package_version="0.1.0",
    )


@pytest.fixture
def tiny_model_path(tmp_path: Path, metadata: ModelMetadata) -> Path:
    """A minimal graph with a dynamic batch axis and metadata attached."""
    graph = helper.make_graph(
        nodes=[
            helper.make_node("ReduceMean", ["input"], ["pooled"], axes=[2, 3], keepdims=0),
            helper.make_node("Concat", ["pooled"] * NUM_CLASSES, ["logits"], axis=1),
        ],
        name="tiny",
        inputs=[
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["batch", 1, IMAGE_SIZE, IMAGE_SIZE]
            )
        ],
        outputs=[
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", NUM_CLASSES])
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)

    for key, value in metadata.to_props().items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value

    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path


def test_props_round_trip(metadata: ModelMetadata) -> None:
    assert ModelMetadata.from_props(metadata.to_props()) == metadata


def test_metadata_survives_onnxruntime(tiny_model_path: Path, metadata: ModelMetadata) -> None:
    """The read path the service uses must recover the write path exactly.

    This is the guard on the whole self-describing-artefact scheme. If it
    fails, the service would fall back to guessed preprocessing parameters and
    serve a model at the wrong resolution without erroring.
    """
    session = ort.InferenceSession(str(tiny_model_path), providers=["CPUExecutionProvider"])
    recovered = ModelMetadata.from_props(session.get_modelmeta().custom_metadata_map)

    assert recovered == metadata
    assert recovered.image_size == 224
    assert recovered.num_classes == NUM_CLASSES


def test_reading_needs_no_onnx_package(tiny_model_path: Path) -> None:
    """Reading goes through onnxruntime alone.

    The serving image ships neither torch nor onnx, so anything the service
    does at startup has to work with onnxruntime by itself.
    """
    session = ort.InferenceSession(str(tiny_model_path), providers=["CPUExecutionProvider"])
    assert ModelMetadata.from_props(session.get_modelmeta().custom_metadata_map)


def test_dynamic_batch_axis(tiny_model_path: Path) -> None:
    """One artefact must serve single requests and calibration batches alike."""
    session = ort.InferenceSession(str(tiny_model_path), providers=["CPUExecutionProvider"])

    for batch in (1, 5):
        data = np.zeros((batch, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        assert session.run(["logits"], {"input": data})[0].shape == (batch, NUM_CLASSES)


def test_missing_keys_are_refused(metadata: ModelMetadata) -> None:
    """A model without preprocessing parameters must not be served on defaults."""
    props = metadata.to_props()
    del props[f"{KEY_PREFIX}image_size"]

    with pytest.raises(ValueError, match="missing metadata keys"):
        ModelMetadata.from_props(props)


def test_unknown_schema_version_is_refused(metadata: ModelMetadata) -> None:
    """An artefact from a newer exporter must fail loudly, not be misread."""
    props = metadata.to_props()
    props[f"{KEY_PREFIX}schema_version"] = "99"

    with pytest.raises(ValueError, match="schema"):
        ModelMetadata.from_props(props)


def test_optional_keys_default(metadata: ModelMetadata) -> None:
    """Provenance fields are nice to have, not required to serve."""
    props = metadata.to_props()
    del props[f"{KEY_PREFIX}git_sha"]

    assert ModelMetadata.from_props(props).git_sha == "unknown"


def test_class_names_keep_their_order(metadata: ModelMetadata) -> None:
    """Index must equal logit index; a reordering would relabel every output."""
    recovered = ModelMetadata.from_props(metadata.to_props())
    assert recovered.class_names == metadata.class_names


def test_summary_mentions_size_and_precision(metadata: ModelMetadata) -> None:
    summary = metadata.summary()
    assert "224px" in summary
    assert "fp32" in summary


# --- External data consolidation -------------------------------------------
# The dynamo exporter writes weights to a .onnx.data sidecar. The artefact has
# to travel as a single release asset, so export folds them back inline and
# removes the sidecar.


@pytest.fixture
def external_data_model(tmp_path: Path, metadata: ModelMetadata) -> Path:
    """A model whose weights live in a separate file, as dynamo emits."""
    from onnx import numpy_helper
    from onnx.external_data_helper import convert_model_to_external_data

    rng = np.random.default_rng(0)
    weight = rng.standard_normal((32, 1, 3, 3)).astype(np.float32)

    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "w"], ["out"], pads=[1, 1, 1, 1])],
        "external",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["batch", 1, IMAGE_SIZE, IMAGE_SIZE]
            )
        ],
        [
            helper.make_tensor_value_info(
                "out", TensorProto.FLOAT, ["batch", 32, IMAGE_SIZE, IMAGE_SIZE]
            )
        ],
        initializer=[numpy_helper.from_array(weight, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    convert_model_to_external_data(
        model, all_tensors_to_one_file=True, location="external.onnx.data", size_threshold=0
    )

    path = tmp_path / "external.onnx"
    onnx.save(model, str(path))
    return path


def test_external_data_files_are_detected(external_data_model: Path) -> None:
    """References must be read without loading the tensors.

    ``onnx.load`` resolves external references and clears them, so inspecting a
    loaded model would always report none and the sidecar would be orphaned
    rather than removed.
    """
    from organ_service.model_meta import external_data_files

    found = external_data_files(external_data_model)
    assert found == {external_data_model.parent / "external.onnx.data"}


def test_attach_metadata_consolidates_artefact(
    external_data_model: Path, metadata: ModelMetadata
) -> None:
    """One file in, one file out, still loadable, metadata readable."""
    from organ_service.model_meta import attach_metadata

    sidecar = external_data_model.parent / "external.onnx.data"
    assert sidecar.exists()

    attach_metadata(external_data_model, metadata)

    assert not sidecar.exists()
    assert external_data_model.stat().st_size > 1000

    session = ort.InferenceSession(str(external_data_model), providers=["CPUExecutionProvider"])
    recovered = ModelMetadata.from_props(session.get_modelmeta().custom_metadata_map)
    assert recovered.image_size == metadata.image_size

    data = np.zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    assert session.run(["out"], {"input": data})[0].shape[1] == 32
