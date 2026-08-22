"""Lightweight preservation and edit-strength metrics for pipeline checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EditMetrics:
    outside_mae: float
    outside_psnr: float
    inside_edit_mae: float
    outside_edge_f1: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_edit(original: Image.Image, edited: Image.Image, matte: np.ndarray) -> EditMetrics:
    source = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
    result = np.asarray(edited.convert("RGB").resize(original.size), dtype=np.float32) / 255.0
    inside = np.clip(matte, 0.0, 1.0)
    outside = 1.0 - inside

    absolute_error = np.abs(result - source).mean(axis=2)
    outside_mae = _weighted_mean(absolute_error, outside)
    outside_mse = _weighted_mean(((result - source) ** 2).mean(axis=2), outside)
    outside_psnr = float(-10.0 * np.log10(max(outside_mse, 1e-10)))
    inside_edit_mae = _weighted_mean(absolute_error, inside)

    source_edges = cv2.Canny((source.mean(axis=2) * 255).astype(np.uint8), 70, 160) > 0
    result_edges = cv2.Canny((result.mean(axis=2) * 255).astype(np.uint8), 70, 160) > 0
    outside_binary = outside > 0.8
    true_positive = np.logical_and(source_edges, result_edges) & outside_binary
    predicted = result_edges & outside_binary
    expected = source_edges & outside_binary
    precision = true_positive.sum() / max(predicted.sum(), 1)
    recall = true_positive.sum() / max(expected.sum(), 1)
    edge_f1 = float(2.0 * precision * recall / max(precision + recall, 1e-10))

    return EditMetrics(
        outside_mae=outside_mae,
        outside_psnr=outside_psnr,
        inside_edit_mae=inside_edit_mae,
        outside_edge_f1=edge_f1,
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float((values * weights).sum() / max(weights.sum(), 1e-10))