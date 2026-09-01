"""Tests for the shared transform.

The transform is the one piece of code that runs in both halves of the system,
so it carries the heaviest test burden. The properties asserted here are the
ones whose violation would degrade the deployed model without failing anything
else: non-determinism, a shape or scale change, or the two entry points
drifting apart.
"""

import numpy as np
import pytest
from PIL import Image

from organ_service import preprocessing as pp


@pytest.fixture
def gradient_image() -> Image.Image:
    """A 28x28 gradient, matching the native OrganAMNIST resolution.

    A gradient rather than noise, so that a resampling change shows up as a
    systematic difference instead of getting lost in the variance.
    """
    data = np.tile(np.arange(28, dtype=np.uint8) * 9, (28, 1))
    return Image.fromarray(data, mode="L")


def test_output_shape_and_dtype(gradient_image: Image.Image) -> None:
    result = pp.preprocess(gradient_image)
    assert result.shape == (pp.NUM_CHANNELS, pp.IMAGE_SIZE, pp.IMAGE_SIZE)
    assert result.dtype == np.float32


def test_is_deterministic(gradient_image: Image.Image) -> None:
    """Repeated calls must be bit-identical.

    Anything stochastic in here would mean the same upload yields different
    predictions on retry, which is the kind of bug users report and nobody
    reproduces.
    """
    first = pp.preprocess(gradient_image)
    second = pp.preprocess(gradient_image)
    np.testing.assert_array_equal(first, second)


def test_normalisation_maps_full_range_symmetrically() -> None:
    """Black maps to -1 and white to +1 under the configured constants."""
    black = pp.preprocess(Image.new("L", (28, 28), color=0))
    white = pp.preprocess(Image.new("L", (28, 28), color=255))

    assert black.max() == pytest.approx(-1.0)
    assert white.min() == pytest.approx(1.0)


def test_rgb_input_is_converted(gradient_image: Image.Image) -> None:
    """An RGB upload of grayscale content must not change the result.

    The browser demo accepts whatever the user drops on it, and PNG exports of
    grayscale images are routinely RGB.
    """
    from_grayscale = pp.preprocess(gradient_image)
    from_rgb = pp.preprocess(gradient_image.convert("RGB"))
    np.testing.assert_allclose(from_grayscale, from_rgb, atol=1e-6)


def test_both_entry_points_agree(gradient_image: Image.Image) -> None:
    """The array path used in training and the image path used in serving
    must produce the same tensor.

    This is the training/serving skew guard. If it ever fails, the deployed
    model is being fed something the trained model never saw.
    """
    array = np.asarray(gradient_image, dtype=np.uint8)
    np.testing.assert_array_equal(
        pp.preprocess(gradient_image),
        pp.preprocess_from_array(array),
    )


def test_channel_last_array_is_accepted(gradient_image: Image.Image) -> None:
    """MedMNIST hands out (H, W, 1); the squeeze must be handled internally."""
    array = np.asarray(gradient_image, dtype=np.uint8)[..., np.newaxis]
    np.testing.assert_array_equal(
        pp.preprocess_from_array(array),
        pp.preprocess(gradient_image),
    )


def test_already_sized_input_is_unscaled() -> None:
    """Values survive when the input is already at target resolution."""
    native = Image.new("L", (pp.IMAGE_SIZE, pp.IMAGE_SIZE), color=128)
    result = pp.preprocess(native)
    expected = (128 / 255.0 - pp.NORM_MEAN) / pp.NORM_STD

    # Absolute tolerance, not relative: the expected value is computed in
    # float64 and lands near zero, where pytest.approx's relative default is
    # tighter than float32 can represent.
    assert result.min() == pytest.approx(expected, abs=1e-6)
    assert result.max() == pytest.approx(expected, abs=1e-6)


def test_load_image_rejects_garbage() -> None:
    """Malformed uploads raise ValueError so the API can answer 4xx, not 500."""
    with pytest.raises(ValueError, match="could not decode"):
        pp.load_image(b"this is not an image")


def test_preprocess_from_array_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="uint8"):
        pp.preprocess_from_array(np.zeros((28, 28), dtype=np.float32))


def test_preprocess_from_array_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match="2D grayscale"):
        pp.preprocess_from_array(np.zeros((28, 28, 3), dtype=np.uint8))


def test_add_batch_axis() -> None:
    sample = np.zeros((1, pp.IMAGE_SIZE, pp.IMAGE_SIZE), dtype=np.float32)
    assert pp.add_batch_axis(sample).shape == (1, 1, pp.IMAGE_SIZE, pp.IMAGE_SIZE)


# --- Affine composition ----------------------------------------------------
# These justify the composed-matrix approach. If the identity case drifts from
# the plain resize path, the matrix is wrong, and every augmented training
# sample is silently misaligned against every inference sample.


def test_identity_params_match_no_params(gradient_image: Image.Image) -> None:
    """``None`` and an explicit identity must be indistinguishable.

    This is the guard on the whole composition scheme: the augmented path with
    zero augmentation has to collapse onto the inference path exactly, not
    approximately.
    """
    np.testing.assert_array_equal(
        pp.preprocess(gradient_image, affine=None),
        pp.preprocess(gradient_image, affine=pp.AffineParams()),
    )


def test_identity_at_native_size_is_lossless() -> None:
    """No resize, no rotation: the pixel values must survive untouched."""
    rng = np.random.default_rng(0)
    source = rng.integers(0, 256, (28, 28), dtype=np.uint8)

    result = pp.preprocess_from_array(source, size=28)
    recovered = (result[0] * pp.NORM_STD + pp.NORM_MEAN) * 255.0

    np.testing.assert_allclose(recovered, source.astype(np.float32), atol=0.51)


@pytest.mark.parametrize("size", [28, 56, 112, 224])
def test_scaling_tracks_pil_resize(size: int) -> None:
    """The composed matrix must reproduce a plain resize up to rounding.

    PIL's resize and transform paths differ by at most one grey level because
    of fixed-point arithmetic, so the tolerance is one step of 1/255 after
    normalisation. A convention error, by contrast, shows up as a shift of
    tens of grey levels, which this catches immediately.
    """
    rng = np.random.default_rng(1)
    source = rng.integers(0, 256, (28, 28), dtype=np.uint8)
    image = Image.fromarray(source, mode="L")

    composed = pp.preprocess(image, size=size)

    resized = np.asarray(image.resize((size, size), pp.RESAMPLE), dtype=np.float32)
    expected = np.expand_dims((resized / 255.0 - pp.NORM_MEAN) / pp.NORM_STD, axis=0)

    # One grey level, plus a touch of slack for float32 representation error.
    tolerance = (1.0 / 255.0) / pp.NORM_STD + 1e-6
    np.testing.assert_allclose(composed, expected, atol=tolerance)


def test_rotation_by_180_flips_both_axes() -> None:
    """A half turn must equal flipping both axes.

    Verifies the rotation centre. An origin-centred rotation, the usual error
    here, would move the content off-canvas entirely.
    """
    rng = np.random.default_rng(2)
    source = rng.integers(0, 256, (28, 28), dtype=np.uint8)

    rotated = pp.preprocess_from_array(source, size=28, affine=pp.AffineParams(rotation_deg=180.0))
    expected = pp.preprocess_from_array(np.rot90(source, 2).copy(), size=28)

    # One grey level of slack: sin(pi) is not exactly zero in floating point,
    # leaving a sub-picometre shear that PIL's fixed-point sampler rounds.
    tolerance = (1.0 / 255.0) / pp.NORM_STD + 1e-6
    np.testing.assert_allclose(rotated, expected, atol=tolerance)


def test_positive_translation_moves_content_right() -> None:
    """Pins the sign convention.

    An inverted sign is geometrically harmless during training but would make
    the documented meaning of the config wrong.
    """
    source = np.zeros((28, 28), dtype=np.uint8)
    source[:, 2] = 255

    shifted = pp.preprocess_from_array(source, size=28, affine=pp.AffineParams(translate_x=0.25))

    # 0.25 of 28 pixels is a 7-pixel shift: the stripe moves from 2 to 9.
    column_means = shifted[0].mean(axis=0)
    assert int(column_means.argmax()) == 9


def test_rotation_fills_corners_with_background() -> None:
    """Pixels pulled in from outside the source get the fill value, not garbage."""
    source = np.full((28, 28), 255, dtype=np.uint8)

    rotated = pp.preprocess_from_array(source, size=28, affine=pp.AffineParams(rotation_deg=45.0))

    background = (pp.FILL_VALUE / 255.0 - pp.NORM_MEAN) / pp.NORM_STD
    assert rotated[0, 0, 0] == pytest.approx(background, abs=1e-6)


def test_augmented_path_is_deterministic(gradient_image: Image.Image) -> None:
    """Augmentation parameters are given, never drawn, so repeats are identical."""
    params = pp.AffineParams(rotation_deg=7.5, translate_x=0.05, translate_y=-0.05)
    np.testing.assert_array_equal(
        pp.preprocess(gradient_image, affine=params),
        pp.preprocess(gradient_image, affine=params),
    )


def test_affine_params_are_frozen() -> None:
    """Sampled parameters must not be mutable after the fact."""
    import dataclasses

    params = pp.AffineParams(rotation_deg=5.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.rotation_deg = 10.0  # type: ignore[misc]
