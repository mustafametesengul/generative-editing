# Selective Weather Editing

Technical assessment submission for structured attribute detection and controllable generative editing. The domain is **weather transformation in outdoor photographs**: changing sky, visibility, precipitation, and surface response while preserving camera, geometry, identities, text, and scene layout.

The pipeline is **generate → verify → select**: FLUX.2 Klein edits the full frame for several seeds, the Task 1 decomposition scores each candidate and hard-gates semantic drift (added people, altered land-water boundaries), and the best passing candidate is returned.

| Before | After (snow) |
|---|---|
| ![Coastal source photograph](artifacts/weather_demo/sources/coastal.jpg) | ![Snow edit: accumulation on land and roofs while the sea stays liquid](artifacts/weather_demo/results/coastal_snow.png) |

More examples, masks, and metrics: [demo README](artifacts/weather_demo/README.md).

## Deliverables

- [Assumptions and domain](docs/ASSUMPTIONS.md)
- [Task 1: perception architecture](docs/TASK1_PERCEPTION.md)
- [Task 2: generative architecture](docs/TASK2_GENERATION.md)
- [Task 3: training and distillation](docs/TASK3_TRAINING_DISTILLATION.md)
- Task 1 prototype: `src/generative_editing/decomposition.py`
- Task 2 prototype: `src/generative_editing/editing.py`, `pipeline.py`, and `evaluation.py`
- [L4 inference demo: sources, edits, masks, metrics, and attribution](artifacts/weather_demo/README.md)

## Setup and checks

```bash
uv sync
uv run python scripts/check_cuda.py
uv run pytest -q
```

The CUDA check performs a real matrix multiplication. The included demo artifacts were generated on an NVIDIA L4.

## Run without model weights

The default path is deterministic and exercises decomposition, editing, candidate verification and selection, metrics, and debug-mask export:

```bash
uv run generative-editing input.jpg output.jpg \
	--weather snow \
	--editor mock \
	--decomposer heuristic \
	--debug-dir artifacts/masks
```

## Run the open-weight model

On a CUDA runtime with at least 13 GiB VRAM:

```bash
uv run generative-editing input.jpg output.jpg \
	--weather rain \
	--editor flux \
	--decomposer transformers \
	--candidates 4 \
	--debug-dir artifacts/masks
```

This lazily downloads `black-forest-labs/FLUX.2-klein-4B`, SegFormer-B2, and Depth Anything V2 Small from Hugging Face. No weights are included in this repository. Add `--cpu-offload` below 13 GiB at a substantial latency cost.

## Hardware recommendation

The selected FLUX.2 Klein 4B model is documented at roughly 13 GiB VRAM, so a 24 GiB L4 is the right inference target. An A100 is not necessary for the proof of concept; use an A100/H100 runtime for LoRA training, distillation, large evaluation batches, or comparison against 20B-class editors.