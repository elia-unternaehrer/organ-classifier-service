"""Command line entry point.

    organ-service download --size 224
    organ-service train --config configs/train_224.yaml
    organ-service export --checkpoint runs/resnet18_224/checkpoint.pt
    organ-service quantize --model artifacts/resnet18_224_fp32.onnx

Subcommands are imported lazily, one at a time. That is not a micro
optimisation: ``train`` pulls in torch and ``download`` pulls in medmnist,
while ``quantize`` needs neither. Importing everything up front would make the
whole command unusable in an environment that has only the artefact
dependencies installed, which is exactly the environment CI runs in.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

COMMANDS = {
    "download": ("organ_service.download", "fetch a dataset and record its provenance"),
    "train": ("organ_service.train", "train a model from a config"),
    "export": ("organ_service.export", "export a checkpoint to ONNX"),
    "quantize": ("organ_service.quantization", "quantise an exported model"),
}


def _load(command: str) -> Callable[[list[str] | None], int]:
    module_name = COMMANDS[command][0]
    module = __import__(module_name, fromlist=["main"])
    return module.main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="organ-service",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS))
    parser.add_argument("-h", "--help", action="store_true")

    known, rest = parser.parse_known_args(argv)

    if known.command is None:
        width = max(len(name) for name in COMMANDS)
        print("usage: organ-service <command> [options]\n\ncommands:")
        for name, (_, description) in sorted(COMMANDS.items()):
            print(f"  {name:<{width}}  {description}")
        print("\nRun 'organ-service <command> --help' for command options.")
        return 0 if known.help else 2

    if known.help:
        rest = [*rest, "--help"]

    return _load(known.command)(rest)


if __name__ == "__main__":
    raise SystemExit(main())
