"""Sampling of geometric augmentations.

The split between this module and ``preprocessing`` is deliberate.
``preprocessing`` *applies* a fully specified transform and stays a pure,
deterministic function that the serving image can import. This module *draws*
one, holds all the randomness, and is never imported at inference time.

Horizontal flipping is absent on purpose. Abdominal organs are not laterally
symmetric, and OrganAMNIST distinguishes ``kidney-left`` from ``kidney-right``
and ``femur-left`` from ``femur-right``. A flip would map a sample onto the
image of a different class while keeping its original label, which is label
noise dressed up as augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from organ_service.preprocessing import IDENTITY, AffineParams


@dataclass(frozen=True)
class AugmentConfig:
    """Bounds of the augmentation distribution.

    Attributes:
        rotation_deg: Rotation is drawn uniformly from ``[-r, +r]``.
        translate: Shift per axis drawn uniformly from ``[-t, +t]``, as a
            fraction of image size.
    """

    rotation_deg: float = 10.0
    translate: float = 0.1

    @property
    def is_identity(self) -> bool:
        return self.rotation_deg == 0.0 and self.translate == 0.0


def sample_affine(config: AugmentConfig, rng: np.random.Generator) -> AffineParams:
    """Draw one augmentation.

    The generator is supplied rather than owned, which is what makes augmented
    training reproducible: the caller derives a generator from
    ``(seed, epoch, index)``, so the transform applied to a given sample in a
    given epoch does not depend on how many dataloader workers happen to be
    running.
    """
    if config.is_identity:
        return IDENTITY

    return AffineParams(
        rotation_deg=float(rng.uniform(-config.rotation_deg, config.rotation_deg)),
        translate_x=float(rng.uniform(-config.translate, config.translate)),
        translate_y=float(rng.uniform(-config.translate, config.translate)),
    )


def generator_for(seed: int, epoch: int, index: int) -> np.random.Generator:
    """Build the per-sample generator.

    Seeding from the tuple rather than advancing a shared stream is what makes
    the result independent of worker count and of iteration order. Two runs
    with the same seed see the same augmentation of sample 17 in epoch 3,
    whether they used one worker or eight.
    """
    return np.random.default_rng([seed, epoch, index])
