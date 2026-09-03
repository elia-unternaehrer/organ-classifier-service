# Three stages, so that a code change does not re-download 67 MB of models and
# a model change does not reinstall the dependency tree.
#
# The runtime image is built from the serving dependency group alone. Torch,
# timm, medmnist, onnx and sympy exist only in the training and dev extras and
# never reach this image: inference runs on ONNX Runtime, and torch alone would
# add roughly 800 MB and put the process past the memory quota of the platforms
# this is meant to run on. That separation is enforced in pyproject.toml and
# guarded by a test, not by discipline here.

ARG PYTHON_VERSION=3.12
ARG MODEL_VERSION=v0.1.0
ARG MODEL_REPO=elia-unternaehrer/organ-classifier-service


# --- models ----------------------------------------------------------------
# Fetched from the GitHub Release rather than copied from the build context, so
# the image is reproducible from a clean checkout and cannot accidentally ship
# a locally modified artefact.

FROM debian:bookworm-slim AS models

ARG MODEL_VERSION
ARG MODEL_REPO

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /models

RUN base="https://github.com/${MODEL_REPO}/releases/download/${MODEL_VERSION}" \
 && for artefact in \
      resnet18_224_fp32.onnx \
      resnet18_224_int8_dynamic.onnx \
      resnet18_224_int8_static.onnx ; do \
      echo "fetching ${artefact}" ; \
      curl -fsSL --retry 3 --retry-delay 2 -o "${artefact}" "${base}/${artefact}" ; \
    done \
 && ls -la /models


# --- builder ---------------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim AS builder

# uv from PyPI rather than from a prebuilt image, so the build needs no
# registry beyond the one it already needs for the base image.
RUN pip install --no-cache-dir uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Dependencies first, in their own layer: they change far less often than the
# source does.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
# --no-editable copies the package into the environment instead of linking back
# to /build, which will not exist in the final image.
RUN uv sync --frozen --no-dev --no-editable


# --- runtime ---------------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim AS runtime

# Unprivileged: nothing here writes to disk, so there is no reason to run as
# root.
RUN useradd --create-home --uid 1000 service

COPY --from=builder --chown=service:service /opt/venv /opt/venv
COPY --from=models --chown=service:service /models /app/models

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models \
    # One thread per operator. On shared-CPU hosting the extra threads contend
    # rather than help, and each costs memory the quota does not have. Raise it
    # on real hardware.
    INTRA_OP_THREADS=1 \
    # ONNX Runtime's arena preallocates generously and does not release between
    # requests, which on a memory-capped platform trades a small speedup for
    # process restarts.
    ENABLE_MEMORY_ARENA=false

USER service
WORKDIR /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ.get('PORT','8000')+'/ready').status==200 else 1)"

# Shell form so ${PORT} expands. Heroku assigns the port at runtime and the
# process must bind to it; locally there is no PORT and 8000 applies.
# A single worker on purpose: models are loaded per process, and a second
# worker would double the resident memory for no throughput a demo needs.
CMD ["sh", "-c", "uvicorn organ_service.serve.app:app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
