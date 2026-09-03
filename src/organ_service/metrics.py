"""Evaluation metrics, defined once.

Balanced accuracy is the selection metric for training, the comparison metric
for quantisation and the headline number in the model card. Those three must
be the same function, not three implementations that happen to agree today.

Argument order follows scikit-learn's ``(y_true, y_pred)`` convention
throughout, since that is what the underlying calls expect and a silent
transposition here would corrupt every number the project reports.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def predictions_from_logits(logits: np.ndarray) -> np.ndarray:
    """Reduce ``(N, C)`` logits to predicted class indices."""
    return np.asarray(logits).argmax(axis=1)


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Macro-averaged recall.

    Chosen over plain accuracy because the class counts are not uniform, and
    over cross-entropy for checkpoint selection because loss weights every
    sample equally and is therefore blind to exactly that imbalance.

    Note that classes absent from ``labels`` are excluded from the average
    rather than counted as zero. On the full splits every class is present, so
    this only matters when evaluating a subset.
    """
    return float(balanced_accuracy_score(labels, predictions))


def per_class_recall(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> list[float]:
    """Recall for each class, in label order.

    Reported in the model card. A single balanced accuracy can hide one class
    performing badly, and for a medical model that is precisely the thing a
    reader needs to be able to check.

    Classes with no support are reported as ``nan`` rather than zero, since no
    recall is defined for them.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    recalls = []
    for klass in range(num_classes):
        mask = labels == klass
        recalls.append(float((predictions[mask] == klass).mean()) if mask.any() else float("nan"))
    return recalls


def confusion(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    """Confusion matrix over the full label range.

    ``labels`` is passed explicitly so the matrix keeps its shape even when a
    class is missing from the evaluated subset; without it the axes would
    silently shift and the plot would mislabel every row.
    """
    return confusion_matrix(labels, predictions, labels=list(range(num_classes)))


def agreement_rate(baseline_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    """Fraction of samples where two models predict the same class.

    The meaningful comparison for a quantised artefact. Its logits are
    guaranteed to differ from the baseline's, so asserting numeric closeness
    answers the wrong question; what a deployment cares about is whether any
    decision changed. A model can shift every logit and still agree on all of
    them.
    """
    return float(
        (
            predictions_from_logits(baseline_logits) == predictions_from_logits(candidate_logits)
        ).mean()
    )
