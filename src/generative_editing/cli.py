"""Command-line entry point for the selective weather editing prototype."""

import json
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg, SettingsConfigDict

from generative_editing.decomposition import (
    HeuristicSceneDecomposer,
    TransformerSceneDecomposer,
    Weather,
)
from generative_editing.editing import Flux2KleinEditor, MockWeatherEditor
from generative_editing.pipeline import WeatherEditingPipeline


class Cli(BaseSettings):
    """Selectively change weather while preserving scene structure."""

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
        env_prefix="GENERATIVE_EDITING_",
    )

    input: CliPositionalArg[Path] = Field(description="Source photograph")
    output: CliPositionalArg[Path] = Field(description="Edited output path")
    weather: Weather
    editor: Literal["mock", "flux"] = "mock"
    decomposer: Literal["heuristic", "transformers"] = "heuristic"
    seed: int = 0
    candidates: int = Field(
        1, ge=1, description="Seeded candidates to generate and verify"
    )
    cpu_offload: bool = Field(
        False, description="Offload FLUX components to CPU to reduce VRAM"
    )
    debug_dir: Path | None = Field(
        None, description="Optional directory for decomposition masks"
    )

    def cli_cmd(self) -> None:
        decomposer = (
            TransformerSceneDecomposer()
            if self.decomposer == "transformers"
            else HeuristicSceneDecomposer()
        )
        editor = (
            Flux2KleinEditor(cpu_offload=self.cpu_offload)
            if self.editor == "flux"
            else MockWeatherEditor()
        )
        pipeline = WeatherEditingPipeline(
            decomposer, editor, candidates=self.candidates
        )

        with Image.open(self.input) as source:
            result = pipeline.run(source, self.weather, seed=self.seed)

        if self.debug_dir:
            result.save_debug_masks(self.debug_dir)
        report = result.metrics.model_dump() | {
            "selected_seed": result.seed,
            "passed_gates": result.passed,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if not result.passed:
            raise SystemExit(
                "No candidate passed the edit and preservation gates; output was not written"
            )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        result.image.save(self.output)


def main() -> None:
    CliApp.run(Cli)
