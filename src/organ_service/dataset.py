"""The torch-facing half of the data pipeline.

Separated from ``organ_service.data`` so that the loading and provenance logic
stays importable without torch, which is what lets CI test it. This module is
training-only and is never imported by the service.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from organ_service.augment import AugmentConfig, generator_for, sample_affine
from organ_service.data import Split
from organ_service.preprocessing import preprocess_from_array


class OrganDataset(Dataset):
    """Applies the shared transform to raw MedMNIST arrays.

    Augmentation is derived from ``(seed, epoch, index)`` rather than drawn
    from a shared stream. That costs one generator construction per sample and
    buys reproducibility that does not depend on ``num_workers``: with a
    shared stream, each worker would consume draws in an interleaved order
    that changes with the worker count.

    Call :meth:`set_epoch` before each epoch. Without it the same augmentation
    repeats every epoch, which silently reduces the effective dataset size.
    """

    def __init__(
        self,
        split: Split,
        image_size: int,
        seed: int,
        augment_config: AugmentConfig | None = None,
    ) -> None:
        self.split = split
        self.image_size = image_size
        self.seed = seed
        self.augment_config = augment_config
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = np.asarray(self.split.images[index])

        affine = None
        if self.augment_config is not None:
            rng = generator_for(self.seed, self.epoch, index)
            affine = sample_affine(self.augment_config, rng)

        array = preprocess_from_array(image, size=self.image_size, affine=affine)
        return torch.from_numpy(array), int(self.split.labels[index])


def build_loader(
    dataset: OrganDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    drop_last: bool = False,
) -> DataLoader:
    """Wrap a dataset in a loader with a seeded shuffle.

    The generator is passed explicitly so that shuffling order is reproducible
    independently of global RNG state, which other libraries are free to
    advance.

    Args:
        drop_last: Discard a trailing partial batch. Required for training:
            BatchNorm cannot compute a variance over a single sample, so a
            leftover batch of one raises at the first norm layer. OrganAMNIST
            hits this exactly, with 34561 training images leaving a remainder
            of one at batch size 128. Leave it off for evaluation, where the
            model is in eval mode and every sample must be scored.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        generator=generator if shuffle else None,
        persistent_workers=num_workers > 0,
    )
