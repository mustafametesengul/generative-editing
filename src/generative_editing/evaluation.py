"""Lightweight preservation and edit-strength metrics for pipeline checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EditMetrics:
    global_edit_mae: float
    semantic_support_mae: float
    weak_support_mae: float
    structure_edge_f1: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_edit(
    original: Image.Image,
    edited: Image.Image,
    matte: np.ndarray,
    structure_guard: np.ndarray | None = None,
) -> EditMetrics:
    source = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
    result = np.asarray(edited.convert("RGB").resize(original.size), dtype=np.float32) / 255.0
    support = np.clip(matte, 0.0, 1.0)
    weak_support = 1.0 - support

    absolute_error = np.abs(result - source).mean(axis=2)
    global_edit_mae = float(absolute_error.mean())
    semantic_support_mae = _weighted_mean(absolute_error, support)
    weak_support_mae = _weighted_mean(absolute_error, weak_support)

    source_edges = cv2.Canny((source.mean(axis=2) * 255).astype(np.uint8), 70, 160) > 0
    result_edges = cv2.Canny((result.mean(axis=2) * 255).astype(np.uint8), 70, 160) > 0
    if structure_guard is None:
        structure_region = cv2.dilate(source_edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    else:
        if structure_guard.shape != source_edges.shape:
            raise ValueError("Structure guard dimensions must match the source image")
        structure_region = cv2.dilate(
            (structure_guard > 0.1).astype(np.uint8), np.ones((5, 5), np.uint8)
        ) > 0
    edge_f1 = _tolerant_edge_f1(source_edges, result_edges, structure_region)

    return EditMetrics(
        global_edit_mae=global_edit_mae,
        semantic_support_mae=semantic_support_mae,
        weak_support_mae=weak_support_mae,
        structure_edge_f1=edge_f1,
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float((values * weights).sum() / max(weights.sum(), 1e-10))


def _tolerant_edge_f1(source: np.ndarray, result: np.ndarray, region: np.ndarray) -> float:
    kernel = np.ones((3, 3), np.uint8)
    source_dilated = cv2.dilate(source.astype(np.uint8), kernel) > 0
    result_dilated = cv2.dilate(result.astype(np.uint8), kernel) > 0
    predicted = result & region
    expected = source & region
    precision = (predicted & source_dilated).sum() / max(predicted.sum(), 1)
    recall = (expected & result_dilated).sum() / max(expected.sum(), 1)
    return float(2.0 * precision * recall / max(precision + recall, 1e-10))