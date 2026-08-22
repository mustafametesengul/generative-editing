"""Command-line entry point for the selective weather editing prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from generative_editing.decomposition import (
    HeuristicSceneDecomposer,
    TransformerSceneDecomposer,
    Weather,
)
from generative_editing.editing import Flux2KleinEditor, MockWeatherEditor
from generative_editing.pipeline import WeatherEditingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Selectively change weather while preserving scene structure")
    parser.add_argument("input", type=Path, help="Source photograph")
    parser.add_argument("output", type=Path, help="Edited output path")
    parser.add_argument("--weather", choices=[weather.value for weather in Weather], required=True)
    parser.add_argument("--editor", choices=("mock", "flux"), default="mock")
    parser.add_argument("--decomposer", choices=("heuristic", "transformers"), default="heuristic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=1, help="Seeded candidates to generate and verify")
    parser.add_argument("--cpu-offload", action="store_true", help="Offload FLUX components to CPU to reduce VRAM")
    parser.add_argument("--debug-dir", type=Path, help="Optional directory for decomposition masks")
    args = parser.parse_args()

    decomposer = (
        TransformerSceneDecomposer()
        if args.decomposer == "transformers"
        else HeuristicSceneDecomposer()
    )
    editor = Flux2KleinEditor(cpu_offload=args.cpu_offload) if args.editor == "flux" else MockWeatherEditor()
    pipeline = WeatherEditingPipeline(decomposer, editor, candidates=args.candidates)

    with Image.open(args.input) as source:
        result = pipeline.run(source, Weather(args.weather), seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(args.output)
    if args.debug_dir:
        result.save_debug_masks(args.debug_dir)
    report = result.metrics.as_dict() | {"selected_seed": result.seed, "passed_preservation": result.passed}
    print(json.dumps(report, indent=2, sort_keys=True))