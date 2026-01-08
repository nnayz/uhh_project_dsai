"""Evaluation helpers for few-shot experiments."""

import numpy as np


def accuracy(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return (preds == labels).mean()


def evaluate_fewshot(preds, labels):
    """Return accuracy and simple stats."""
    acc = accuracy(preds, labels)
    return {"accuracy": float(acc)}
