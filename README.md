# Organ Classifier Service

Organ classification on abdominal CT slices, served as a containerised ONNX
inference API with a browser demo.

> **Status:** scaffolding. Training, export and serving land next.

<!-- TODO: live demo link -->
<!-- TODO: frontend screenshot -->

## Results

<!-- TODO: balanced accuracy, per-class recall, confusion matrix -->

## Quickstart

```bash
docker compose up
# open http://localhost:8000
```

## Development

```bash
uv sync --group dev          # serving deps + tooling
uv sync --extra train        # adds torch, timm, medmnist for training
uv run pytest -m "not slow"
uv run ruff check .
```

## Architecture

```
                  ┌─────────────────────────────┐
  browser  ───▶   │  FastAPI                    │
                  │   /predict  /health         │
                  │   /ready    /metrics        │
                  │                             │
                  │   ONNX Runtime session      │
                  └─────────────────────────────┘
                            ▲
                            │  model.onnx (GitHub Release asset)
```

Training and serving both import `organ_service.preprocessing`. The transform
is defined once, so the tensor produced at inference time is identical to the
one seen during training. Training/serving skew is the most common way an
image classifier degrades silently in production, and defining the transform in
two places is how it usually happens.

## Design decisions

Each entry records the decision, the reason, and what would change it.

<!-- TODO: no Kubernetes / no database / no model registry / no monitoring
     stack / ONNX instead of torch at serving time -->

## Limitations

<!-- TODO -->

## Licence

Code is MIT. Dataset licensing is documented in `MODEL_CARD.md`.
