"""Prometheus instrumentation.

Named telemetry rather than metrics to keep it apart from
``organ_service.metrics``, which holds evaluation metrics. Two modules called
metrics with unrelated meanings would be a small trap in every future import.

Deliberately minimal. The point is not to operate this service at scale but to
expose the counters that would matter if someone did: how many predictions,
split by model and outcome, and how long each stage took. A full monitoring
stack is a deployment concern and belongs outside the image; what belongs
inside is an endpoint worth scraping.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Buckets in seconds, chosen for this workload rather than left at the default.
# ResNet18 at 224 pixels on a shared CPU lands somewhere around 100 ms, so the
# resolution needs to be fine between 10 and 500 ms; the default buckets spend
# most of their range above one second, where nothing will ever land.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.5)


class Telemetry:
    """Collectors for one service instance.

    Bound to an explicit registry rather than the process-global default, so
    that tests can build an instance without leaking collectors into every
    subsequent test in the session.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()

        self.predictions = Counter(
            "organ_predictions_total",
            "Prediction requests served",
            labelnames=("model_id", "outcome"),
            registry=self.registry,
        )
        self.stage_seconds = Histogram(
            "organ_prediction_stage_seconds",
            "Time spent per prediction stage",
            labelnames=("model_id", "stage"),
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.model_info = Gauge(
            "organ_model_loaded",
            "Loaded artefacts, labelled with their provenance",
            labelnames=("model_id", "precision", "image_size", "model_version"),
            registry=self.registry,
        )

    def record_models(self, described: list[dict]) -> None:
        """Publish one series per loaded artefact.

        Makes the deployed model version visible to whatever scrapes this,
        which is the question that actually gets asked during an incident.
        """
        for entry in described:
            self.model_info.labels(
                model_id=entry["model_id"],
                precision=entry["precision"],
                image_size=str(entry["image_size"]),
                model_version=entry["model_version"],
            ).set(1)

    def record_success(
        self, model_id: str, preprocess_seconds: float, inference_seconds: float
    ) -> None:
        self.predictions.labels(model_id=model_id, outcome="success").inc()
        self.stage_seconds.labels(model_id=model_id, stage="preprocess").observe(preprocess_seconds)
        self.stage_seconds.labels(model_id=model_id, stage="inference").observe(inference_seconds)

    def record_failure(self, model_id: str, outcome: str) -> None:
        """Count rejected requests.

        Labelled by cause rather than lumped into one error counter: a spike in
        oversized uploads and a spike in undecodable files call for different
        responses.
        """
        self.predictions.labels(model_id=model_id, outcome=outcome).inc()
