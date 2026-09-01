"""Deterministic image preprocessing shared by training and serving.

This module is the single definition of the transform that turns an image into
model input. Both halves of the project import it: the training pipeline wraps
the result in a torch tensor, the inference service adds a batch axis and hands
it to ONNX Runtime.

Three constraints keep that guarantee intact, and all three are load-bearing.

**No torch.** Only PIL and NumPy are imported here. The serving image ships
ONNX Runtime and nothing else, so a torch import would make this module
unloadable there and force a second implementation of the same transform.

**PIL on both sides.** ``torchvision.transforms`` and PIL do not resample
identically, and training on one while serving on the other introduces a
distribution shift that no metric explains. PIL is the only imaging library
present in both environments.

**One resampling step.** Resize and augmentation are composed into a single
affine matrix and applied once, rather than resizing and then rotating. Two
successive bilinear passes blur twice, and on 28x28 source images that is not
a rounding detail. Composition also removes an arbitrary choice, since
rotating before upsampling and rotating after it are not the same operation.

Randomness deliberately lives elsewhere. This module *applies* a transform it
is fully given; ``organ_service.augment`` is what *samples* one. That keeps
``preprocess`` a pure function whose determinism is testable.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

# --- Transform constants ---------------------------------------------------
# Changing any of these invalidates every exported model, because the tensor
# the network sees changes shape or scale. They travel with the model as ONNX
# metadata so that the service configures itself from the artefact it loads.

IMAGE_SIZE = 224
"""Default square side length in pixels fed to the network.

OrganAMNIST is natively 28x28. The resolution ablation trains at 28, 56, 112
and 224, so callers normally pass this explicitly; the constant is the
fallback for the pretrained-backbone default.
"""

RESAMPLE = Image.Resampling.BILINEAR
"""Pinned explicitly. PIL's default has changed across major versions."""

FILL_VALUE = 0
"""Value for pixels pulled in from outside the source under rotation.

Black matches the background of the abdominal CT windowing used by MedMNIST,
so rotated corners do not introduce a brightness artefact the network could
learn from.
"""

NORM_MEAN = 0.5
NORM_STD = 0.5
"""Normalisation constants used by the MedMNIST reference implementation.

Keeping them makes results directly comparable to the published baselines,
which is worth more here than dataset-specific statistics. Run
``scripts/compute_norm_stats.py`` to derive the empirical values instead.
"""

NUM_CHANNELS = 1


@dataclass(frozen=True)
class AffineParams:
    """A concrete geometric augmentation, already sampled.

    Frozen and fully specified: no distributions, no RNG. The training loop
    draws these per sample and hands them in; the service never constructs one.
    The default value is the identity, which makes ``AffineParams()`` and
    ``None`` interchangeable.

    Attributes:
        rotation_deg: Counter-clockwise rotation about the image centre.
        translate_x: Horizontal shift as a fraction of output width. Positive
            moves image content to the right.
        translate_y: Vertical shift as a fraction of output height. Positive
            moves image content downward.
    """

    rotation_deg: float = 0.0
    translate_x: float = 0.0
    translate_y: float = 0.0


IDENTITY = AffineParams()


def _affine_matrix(
    src_width: int,
    src_height: int,
    size: int,
    params: AffineParams,
) -> tuple[float, float, float, float, float, float]:
    """Compose scaling, rotation and translation into one PIL affine matrix.

    PIL's ``Image.transform`` expects the *inverse* mapping: for each output
    pixel it evaluates the matrix to find where to sample in the source. It
    also works in continuous coordinates, substituting ``x + 0.5`` for the
    pixel index itself, so the half-pixel convention must not be applied here a
    second time. Getting that wrong shifts the whole image by a quarter pixel
    per axis, which is invisible by eye and quietly degrades training.

    The forward transform being inverted is: scale the source up to ``size``,
    rotate about the centre of the output, then translate.

    Returns:
        The six coefficients ``(a, b, c, d, e, f)`` such that a source
        coordinate is ``(a*X + b*Y + c, d*X + e*Y + f)``.
    """
    scale_x = size / src_width
    scale_y = size / src_height

    theta = math.radians(params.rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    centre = size / 2.0
    shift_x = params.translate_x * size
    shift_y = params.translate_y * size

    offset_x = -shift_x - centre
    offset_y = -shift_y - centre

    return (
        cos_t / scale_x,
        sin_t / scale_x,
        (cos_t * offset_x + sin_t * offset_y + centre) / scale_x,
        -sin_t / scale_y,
        cos_t / scale_y,
        (-sin_t * offset_x + cos_t * offset_y + centre) / scale_y,
    )


def load_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes into a PIL image.

    Kept separate from :func:`preprocess` so that the training pipeline, which
    already holds decoded arrays, does not pay for a round trip through an
    encoder.

    Raises:
        ValueError: If the bytes are not a decodable image. Callers surface
            this as a 4xx rather than letting it become a 500.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        # Any decode failure is a bad request, not a server error.
        raise ValueError(f"could not decode image: {exc}") from exc
    return image


def preprocess(
    image: Image.Image,
    size: int = IMAGE_SIZE,
    affine: AffineParams | None = None,
) -> np.ndarray:
    """Apply the deterministic transform.

    Resizing always goes through the affine path, including when ``affine`` is
    ``None``. Using ``Image.resize`` in that case would be the obvious
    shortcut, but PIL's resize and transform code paths differ by up to one
    grey level due to fixed-point rounding, which would make the augmented and
    unaugmented paths subtly incomparable. One code path removes the question.

    Args:
        image: Any PIL image. Mode, orientation and size are normalised here,
            so callers do not need to pre-condition browser uploads.
        size: Square side length of the output.
        affine: A sampled geometric augmentation, or ``None`` for the identity.
            Training passes one per sample; inference always passes ``None``.

    Returns:
        Array of shape ``(NUM_CHANNELS, size, size)``, dtype float32,
        standardised to roughly zero mean and unit variance. No batch axis.
    """
    # Browser uploads carry EXIF orientation; a rotated input would otherwise
    # reach the network rotated.
    image = ImageOps.exif_transpose(image)

    # Uploads may arrive as RGB, RGBA or palette-based. The network takes one
    # channel, and PIL's L conversion applies the standard luma weights.
    if image.mode != "L":
        image = image.convert("L")

    params = IDENTITY if affine is None else affine
    matrix = _affine_matrix(image.width, image.height, size, params)
    image = image.transform(
        (size, size),
        Image.Transform.AFFINE,
        matrix,
        resample=RESAMPLE,
        fillcolor=FILL_VALUE,
    )

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - NORM_MEAN) / NORM_STD

    # (H, W) -> (C, H, W)
    return np.expand_dims(array, axis=0)


def preprocess_from_array(
    array: np.ndarray,
    size: int = IMAGE_SIZE,
    affine: AffineParams | None = None,
) -> np.ndarray:
    """Apply the transform to a raw uint8 array.

    The training pipeline reads MedMNIST as a stack of uint8 arrays and never
    touches an image file. Routing it through PIL here rather than
    reimplementing the resize in NumPy is the point: the resampling code path
    stays identical to the one used at inference time.

    Args:
        array: ``(H, W)`` or ``(H, W, 1)``, dtype uint8.
        size: Square side length of the output.
        affine: A sampled geometric augmentation, or ``None`` for the identity.

    Returns:
        Array of shape ``(NUM_CHANNELS, size, size)``, dtype float32.
    """
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"expected a 2D grayscale array, got shape {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"expected dtype uint8, got {array.dtype}")

    return preprocess(Image.fromarray(array, mode="L"), size=size, affine=affine)


def add_batch_axis(sample: np.ndarray) -> np.ndarray:
    """Turn ``(C, H, W)`` into the ``(1, C, H, W)`` the ONNX graph expects."""
    return np.expand_dims(sample, axis=0)
