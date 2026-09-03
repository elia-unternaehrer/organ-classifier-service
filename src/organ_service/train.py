"""Train the classifier.

    uv run --extra train python -m organ_service.train --config configs/train.yaml

The test split is never touched here. Selection across runs happens on
validation balanced accuracy alone, and ``evaluate.py`` reads the test split
once, after the choice has been committed. Keeping the two in separate entry
points is what makes that claim checkable rather than merely stated: the
commit fixing the selection precedes the commit carrying test numbers.
"""

from __future__ import annotations

import os

# cuBLAS needs a fixed workspace before its first handle is created for
# deterministic reductions, so this is set ahead of importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
import yaml
from torch import nn

from organ_service import __version__
from organ_service.augment import AugmentConfig
from organ_service.data import (
    class_distribution,
    load_manifest,
    load_split,
)
from organ_service.dataset import OrganDataset, build_loader
from organ_service.metrics import balanced_accuracy, confusion
from organ_service.preprocessing import NORM_MEAN, NORM_STD

# Binary datasets are the two-class case of the same setup and need no
# special handling.
SUPPORTED_TASKS = {"multi-class", "binary-class"}

# --- Configuration ---------------------------------------------------------


@dataclass(frozen=True)
class Config:
    run_name: str
    seed: int
    dataset: str
    image_size: int
    data_root: Path
    num_workers: int
    mmap: bool
    model_name: str
    pretrained: bool
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    amp: str
    augment: AugmentConfig

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        raw = yaml.safe_load(path.read_text())
        data, model, train, aug = (
            raw["data"],
            raw["model"],
            raw["train"],
            raw.get("augment", {}),
        )

        if train["amp"] not in {"off", "fp16", "bf16"}:
            raise ValueError(f"amp must be off, fp16 or bf16, got {train['amp']!r}")
        if train["warmup_epochs"] >= train["epochs"]:
            raise ValueError("warmup_epochs must be shorter than epochs")

        return cls(
            run_name=raw["run_name"],
            seed=raw["seed"],
            dataset=data["dataset"],
            image_size=data["size"],
            data_root=Path(data["root"]),
            num_workers=data["num_workers"],
            mmap=data.get("mmap", False),
            model_name=model["name"],
            pretrained=model["pretrained"],
            epochs=train["epochs"],
            batch_size=train["batch_size"],
            lr=float(train["lr"]),
            weight_decay=float(train["weight_decay"]),
            warmup_epochs=train["warmup_epochs"],
            amp=train["amp"],
            augment=AugmentConfig(
                rotation_deg=float(aug.get("rotation_deg", 0.0)),
                translate=float(aug.get("translate", 0.0)),
            ),
        )

    def to_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items()}
        out["data_root"] = str(self.data_root)
        out["augment"] = {
            "rotation_deg": self.augment.rotation_deg,
            "translate": self.augment.translate,
        }
        return out


# --- Reproducibility -------------------------------------------------------


def set_determinism(seed: int) -> None:
    """Pin every RNG and disable nondeterministic kernels.

    ``cudnn.benchmark`` is switched off deliberately. It picks convolution
    algorithms by timing them, so the choice, and with it the numerics, can
    differ between otherwise identical runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_sha() -> str:
    """Record the commit that produced a run, or note that it is unknown."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# --- Training --------------------------------------------------------------


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: Config, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup into a cosine decay, stepped per batch."""
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def autocast_context(amp: str, device: torch.device):
    if amp == "off" or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    amp: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """One pass. Training when an optimizer is supplied, evaluation otherwise."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_seen = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast_context(amp, device):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            batch = labels.size(0)
            total_loss += loss.item() * batch
            total_seen += batch
            predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    return (
        total_loss / max(total_seen, 1),
        np.concatenate(predictions),
        np.concatenate(targets),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)

    config = Config.from_yaml(args.config)
    set_determinism(config.seed)

    run_dir = args.out / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"run {config.run_name} on {device}")

    provenance = load_manifest(config.dataset, config.image_size, config.data_root)
    num_classes = provenance.num_classes

    # The pipeline assumes single-label classification end to end: softmax over
    # logits, cross-entropy, argmax, balanced accuracy. Several MedMNIST
    # datasets are not that. ChestMNIST is multi-label and needs BCE with a
    # sigmoid head and AUC in place of balanced accuracy; RetinaMNIST is
    # ordinal. Failing here marks the branch point explicitly rather than
    # training something that reports plausible but meaningless numbers.
    if provenance.task not in SUPPORTED_TASKS:
        raise SystemExit(
            f"{config.dataset} is a '{provenance.task}' task; this pipeline "
            f"supports {sorted(SUPPORTED_TASKS)}. Supporting it would mean a "
            f"different loss, head activation and evaluation metric."
        )

    splits = {
        name: load_split(
            config.dataset, config.image_size, config.data_root, name, mmap=config.mmap
        )
        # The test split is intentionally absent.
        for name in ("train", "val")
    }

    train_ds = OrganDataset(splits["train"], config.image_size, config.seed, config.augment)
    val_ds = OrganDataset(splits["val"], config.image_size, config.seed, None)

    train_loader = build_loader(
        train_ds,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        # OrganAMNIST's 34561 training images leave a remainder of one at the
        # default batch size, and BatchNorm cannot train on a single sample.
        drop_last=True,
    )
    val_loader = build_loader(
        val_ds,
        config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    model = timm.create_model(
        config.model_name,
        pretrained=config.pretrained,
        num_classes=num_classes,
        # timm folds the pretrained RGB stem down to one channel by summing
        # across the input axis, so the ImageNet initialisation survives.
        in_chans=1,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=(config.amp == "fp16"))

    history_path = run_dir / "history.csv"
    history_file = history_path.open("w", newline="")
    writer = csv.writer(history_file)
    writer.writerow(["epoch", "lr", "train_loss", "train_bacc", "val_loss", "val_bacc", "seconds"])

    best_bacc = -1.0
    best_loss = float("inf")
    best_epoch = -1
    best_confusion: np.ndarray | None = None

    # Validation runs after every training epoch; each row below is one pass
    # over train followed by one over val. The trailing asterisk marks epochs
    # where the checkpoint was replaced.
    print(
        f"\n{'epoch':>5}  {'lr':>9}  {'train_loss':>10} {'train_bacc':>10}  "
        f"{'val_loss':>8} {'val_bacc':>8}  {'secs':>6}"
    )
    print("-" * 68)

    for epoch in range(config.epochs):
        started = time.perf_counter()
        train_ds.set_epoch(epoch)

        train_loss, train_pred, train_true = run_epoch(
            model, train_loader, criterion, device, config.amp, optimizer, scheduler, scaler
        )
        val_loss, val_pred, val_true = run_epoch(model, val_loader, criterion, device, config.amp)

        train_bacc = balanced_accuracy(train_true, train_pred)
        val_bacc = balanced_accuracy(val_true, val_pred)
        elapsed = time.perf_counter() - started

        writer.writerow(
            [
                epoch,
                f"{scheduler.get_last_lr()[0]:.6e}",
                f"{train_loss:.6f}",
                f"{train_bacc:.6f}",
                f"{val_loss:.6f}",
                f"{val_bacc:.6f}",
                f"{elapsed:.1f}",
            ]
        )
        history_file.flush()

        # Selection on the target metric, not on loss. Cross-entropy weights
        # every sample equally and so is blind to the class imbalance that
        # balanced accuracy exists to account for; the two also diverge late in
        # training when the network grows overconfident without changing its
        # decisions. Loss breaks ties only.
        improved = val_bacc > best_bacc or (val_bacc == best_bacc and val_loss < best_loss)
        if improved:
            best_bacc, best_loss, best_epoch = val_bacc, val_loss, epoch
            best_confusion = confusion(val_true, val_pred, num_classes)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_balanced_accuracy": val_bacc,
                    "val_loss": val_loss,
                    "config": config.to_dict(),
                    "num_classes": num_classes,
                    "class_names": provenance.class_names,
                    # Carried so that ONNX export needs no second source for
                    # the transform parameters, which could drift from these.
                    "preprocessing": {
                        "image_size": config.image_size,
                        "norm_mean": NORM_MEAN,
                        "norm_std": NORM_STD,
                    },
                },
                run_dir / "checkpoint.pt",
            )

        marker = " *" if improved else "  "
        print(
            f"{epoch:5d}  {scheduler.get_last_lr()[0]:9.2e}  "
            f"{train_loss:10.4f} {train_bacc:10.4f}  "
            f"{val_loss:8.4f} {val_bacc:8.4f}  {elapsed:6.1f}{marker}"
        )

    history_file.close()

    metrics = {
        "run_name": config.run_name,
        "package_version": __version__,
        "git_sha": git_sha(),
        "selection": {
            "metric": "validation balanced accuracy",
            "best_epoch": best_epoch,
            "val_balanced_accuracy": best_bacc,
            "val_loss": best_loss,
            "note": "test split not read by this script",
        },
        "config": config.to_dict(),
        "data": provenance.to_dict()
        | {
            "class_distribution": {
                name: class_distribution(split, num_classes) for name, split in splits.items()
            }
        },
        "environment": {
            "torch": torch.__version__,
            "timm": timm.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "confusion_matrix_val": (best_confusion.tolist() if best_confusion is not None else None),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\nbest epoch {best_epoch}, val balanced accuracy {best_bacc:.4f}")
    print(f"wrote {run_dir}/checkpoint.pt, metrics.json, history.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
