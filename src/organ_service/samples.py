"""Export sample slices for the browser demo.

    organ-service samples --size 224

Nobody evaluating this repository has an abdominal CT slice on their desktop.
Without shipped examples the demo is technically working and practically dead,
so the frontend gets a strip of real slices to click.

Samples are drawn from the validation split. They are illustrations, not
measurements, but the test split stays untouched on principle: the one rule
this project makes about it is easier to keep than to qualify.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from organ_service.data import load_manifest, load_split

DEFAULT_OUTPUT = Path("src/organ_service/serve/static/samples")
DEFAULT_COUNT = 6


def choose_indices(labels: np.ndarray, count: int, seed: int) -> list[int]:
    """Pick samples from distinct classes.

    One per class rather than at random: six random slices from an unbalanced
    split would likely show three livers, and a demo whose examples all
    predict the same class demonstrates nothing.
    """
    rng = np.random.default_rng(seed)
    classes = rng.permutation(np.unique(labels))[:count]

    return [int(rng.choice(np.flatnonzero(labels == klass))) for klass in classes]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="organamnist")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    provenance = load_manifest(args.dataset, args.size, args.data_root)
    split = load_split(args.dataset, args.size, args.data_root, "val")

    args.out.mkdir(parents=True, exist_ok=True)
    for stale in args.out.glob("*.png"):
        stale.unlink()

    entries = []
    for index in choose_indices(split.labels, args.count, args.seed):
        label = int(split.labels[index])
        name = provenance.class_names[label]
        filename = f"{name.replace(' ', '-')}.png"

        Image.fromarray(np.asarray(split.images[index]), mode="L").save(args.out / filename)
        entries.append(
            {
                "file": filename,
                "label_index": label,
                "label_name": name,
                # Carried so the demo can show what the answer should be, which
                # is what makes a correct prediction legible to a visitor who
                # cannot read a CT slice.
                "source": f"{args.dataset} val[{index}]",
            }
        )

    manifest = {
        "dataset": args.dataset,
        "size": args.size,
        "split": "val",
        "seed": args.seed,
        "license": provenance.license,
        "samples": entries,
    }
    (args.out / "samples.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {len(entries)} samples to {args.out}")
    for entry in entries:
        print(f"  {entry['file']:<20} {entry['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
