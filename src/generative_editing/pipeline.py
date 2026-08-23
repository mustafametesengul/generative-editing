"""End-to-end selective weather editing pipeline.

The generator owns every output pixel. The Task 1 decomposition is a
verification contract: it scores where change is licensed, flags leakage
elsewhere, and selects the best seeded candidate.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict

from generative_editing.decomposition import (
    SceneDecomposer,
    SceneDecomposition,
    Weather,
)
from generative_editing.editing import ImageEditor
from generative_editing.evaluation import (
    EditMetrics,
    candidate_score,
    evaluate_edit,
    passes_gates,
)


class PipelineResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    image: Image.Image
    decomposition: SceneDecomposition
    matte: np.ndarray
    generation_matte: np.ndarray
    metrics: EditMetrics
    seed: int
    passed: bool

    def save_debug_masks(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        masks = {
            "sky": self.decomposition.sky,
            "atmosphere": self.decomposition.atmosphere,
            "weather_surface": self.decomposition.weather_surface,
            "structure_guard": self.decomposition.structure_guard,
            "confidence": self.decomposition.confidence,
            "edit_matte": self.matte,
            "generation_matte": self.generation_matte,
        }
        for name, mask in masks.items():
            Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)).save(
                directory / f"{name}.png"
            )


class WeatherEditingPipeline:
    def __init__(
        self, decomposer: SceneDecomposer, editor: ImageEditor, candidates: int = 1
    ) -> None:
        if candidates < 1:
            raise ValueError("At least one candidate is required")
        self.decomposer = decomposer
        self.editor = editor
        self.candidates = candidates

    def run(
        self, image: Image.Image, weather: Weather, seed: int = 0
    ) -> PipelineResult:
        source = image.convert("RGB")
        decomposition = self.decomposer.decompose(source)
        source_layout = self.decomposer.layout(source)
        matte = decomposition.edit_matte(weather)
        generation_matte = decomposition.generation_matte(weather)

        best: PipelineResult | None = None
        best_key: tuple[bool, float] = (False, float("-inf"))
        for offset in range(self.candidates):
            candidate_seed = seed + offset
            candidate = self.editor.edit(source, weather, candidate_seed)
            metrics = evaluate_edit(
                source,
                candidate,
                matte,
                decomposition.structure_guard,
                source_layout,
                self.decomposer.layout(candidate),
            )
            passed = passes_gates(metrics)
            key = (passed, candidate_score(metrics))
            if best is None or key > best_key:
                best = PipelineResult(
                    image=candidate,
                    decomposition=decomposition,
                    matte=matte,
                    generation_matte=generation_matte,
                    metrics=metrics,
                    seed=candidate_seed,
                    passed=passed,
                )
                best_key = key
        assert best is not None
        return best
