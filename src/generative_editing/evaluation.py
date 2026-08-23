"""Lightweight preservation and edit-strength metrics for pipeline checks."""

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict

from generative_editing.decomposition import SceneLayout


class EditMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    global_edit_mae: float
    semantic_support_mae: float
    weak_support_mae: float
    structure_edge_f1: float
    protected_gain: float
    water_loss: float
    water_gain: float


def evaluate_edit(
    original: Image.Image,
    edited: Image.Image,
    matte: np.ndarray,
    structure_guard: np.ndarray | None = None,
    source_layout: SceneLayout | None = None,
    candidate_layout: SceneLayout | None = None,
) -> EditMetrics:
    source = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
    result = (
        np.asarray(edited.convert("RGB").resize(original.size), dtype=np.float32)
        / 255.0
    )
    support = np.clip(matte, 0.0, 1.0)
    weak_support = 1.0 - support

    absolute_error = np.abs(result - source).mean(axis=2)
    global_edit_mae = float(absolute_error.mean())
    semantic_support_mae = _weighted_mean(absolute_error, support)
    weak_support_mae = _weighted_mean(absolute_error, weak_support)

    source_edges = cv2.Canny((source.mean(axis=2) * 255).astype(np.uint8), 70, 160) > 0
    result_edges = cv2.Canny((result.mean(axis=2) * 255).astype(np.uint8), 70, 160) > 0
    if structure_guard is None:
        structure_region = (
            cv2.dilate(source_edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        )
    else:
        if structure_guard.shape != source_edges.shape:
            raise ValueError("Structure guard dimensions must match the source image")
        structure_region = (
            cv2.dilate(
                (structure_guard > 0.1).astype(np.uint8), np.ones((5, 5), np.uint8)
            )
            > 0
        )
    edge_f1 = _tolerant_edge_f1(source_edges, result_edges, structure_region)

    protected_gain = 0.0
    water_loss = 0.0
    water_gain = 0.0
    if source_layout is not None and candidate_layout is not None:
        protected_gain, water_loss, water_gain = _layout_drift(
            source_layout, candidate_layout
        )

    return EditMetrics(
        global_edit_mae=global_edit_mae,
        semantic_support_mae=semantic_support_mae,
        weak_support_mae=weak_support_mae,
        structure_edge_f1=edge_f1,
        protected_gain=protected_gain,
        water_loss=water_loss,
        water_gain=water_gain,
    )


def _layout_drift(
    source: SceneLayout, candidate: SceneLayout, minimum_water: float = 0.005
) -> tuple[float, float, float]:
    """Appeared protected content, source water turned solid, and new water over solid ground."""
    protected_gain = float((candidate.protected & ~source.protected).mean())
    water_gain = float((candidate.water & ~source.water & ~source.sky).mean())
    water_area = float(source.water.mean())
    if water_area < minimum_water:
        return protected_gain, 0.0, water_gain
    flipped = source.water & ~candidate.water & ~candidate.sky
    return protected_gain, float(flipped.mean() / water_area), water_gain


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float((values * weights).sum() / max(weights.sum(), 1e-10))


def candidate_score(metrics: EditMetrics, minimum_edit: float = 0.02) -> float:
    """Rank candidates by preservation once a visible edit is confirmed."""
    if metrics.semantic_support_mae < minimum_edit:
        return float("-inf")
    return (
        metrics.structure_edge_f1
        - 2.0 * metrics.weak_support_mae
        - 50.0 * metrics.protected_gain
        - 8.0 * metrics.water_loss
        - 8.0 * metrics.water_gain
    )


def passes_gates(
    metrics: EditMetrics,
    minimum_edit: float = 0.02,
    max_protected_gain: float = 0.001,
    max_water_loss: float = 0.03,
    max_water_gain: float = 0.02,
) -> bool:
    """Hard gates for visible editing and semantic preservation."""
    return (
        metrics.semantic_support_mae >= minimum_edit
        and metrics.protected_gain <= max_protected_gain
        and metrics.water_loss <= max_water_loss
        and metrics.water_gain <= max_water_gain
    )


def _tolerant_edge_f1(
    source: np.ndarray, result: np.ndarray, region: np.ndarray
) -> float:
    kernel = np.ones((3, 3), np.uint8)
    source_dilated = cv2.dilate(source.astype(np.uint8), kernel) > 0
    result_dilated = cv2.dilate(result.astype(np.uint8), kernel) > 0
    predicted = result & region
    expected = source & region
    precision = (predicted & source_dilated).sum() / max(predicted.sum(), 1)
    recall = (expected & result_dilated).sum() / max(expected.sum(), 1)
    return float(2.0 * precision * recall / max(precision + recall, 1e-10))
