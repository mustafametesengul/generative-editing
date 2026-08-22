"""End-to-end selective weather editing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from generative_editing.decomposition import SceneDecomposer, SceneDecomposition, Weather
from generative_editing.editing import ImageEditor, selective_composite
from generative_editing.evaluation import EditMetrics, evaluate_edit


@dataclass(frozen=True)
class PipelineResult:
    image: Image.Image
    decomposition: SceneDecomposition
    matte: np.ndarray
    metrics: EditMetrics

    def save_debug_masks(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        masks = {
            "sky": self.decomposition.sky,
            "atmosphere": self.decomposition.atmosphere,
            "weather_surface": self.decomposition.weather_surface,
            "structure_guard": self.decomposition.structure_guard,
            "confidence": self.decomposition.confidence,
            "edit_matte": self.matte,
        }
        for name, mask in masks.items():
            Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)).save(directory / f"{name}.png")


class WeatherEditingPipeline:
    def __init__(self, decomposer: SceneDecomposer, editor: ImageEditor) -> None:
        self.decomposer = decomposer
        self.editor = editor

    def run(self, image: Image.Image, weather: Weather, seed: int = 0) -> PipelineResult:
        source = image.convert("RGB")
        decomposition = self.decomposer.decompose(source)
        candidate = self.editor.edit(source, weather, seed)
        matte = decomposition.edit_matte(weather)
        output = selective_composite(source, candidate, matte)
        metrics = evaluate_edit(source, output, matte)
        return PipelineResult(output, decomposition, matte, metrics)