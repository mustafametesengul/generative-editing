# Selective Weather Editing

Technical assessment submission for structured attribute detection and controllable generative editing. The chosen domain is **weather transformation in outdoor photographs**: change sky, visibility, precipitation, and weather-dependent surface appearance while preserving camera, geometry, object identity, text, and scene layout.

## Deliverables

- [Assumptions and domain](docs/ASSUMPTIONS.md)
- [Task 1: perception architecture](docs/TASK1_PERCEPTION.md)
- [Task 2: generative architecture](docs/TASK2_GENERATION.md)
- [Task 3: training and distillation](docs/TASK3_TRAINING_DISTILLATION.md)
- Task 1 prototype: `src/generative_editing/decomposition.py`
- Task 2 prototype: `src/generative_editing/editing.py`, `pipeline.py`, and `evaluation.py`

## Setup and checks

```bash
uv sync
uv run python scripts/check_cuda.py
uv run pytest -q
```

The CUDA check performs a real matrix multiplication. At submission time this workspace did not expose an NVIDIA device, although the expected target is an L4.

## Run without model weights

The default path is deterministic and exercises decomposition, editing, selective compositing, metrics, and debug-mask export:

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
	--debug-dir artifacts/masks
```

This lazily downloads `black-forest-labs/FLUX.2-klein-4B`, SegFormer-B2, and Depth Anything V2 Small from Hugging Face. No weights are included in this repository. Add `--cpu-offload` below 13 GiB at a substantial latency cost.

## Hardware recommendation

The selected FLUX.2 Klein 4B model is documented at roughly 13 GiB VRAM, so a 24 GiB L4 is the right inference target. An A100 is not necessary for the proof of concept; use an A100/H100 runtime for LoRA training, distillation, large evaluation batches, or comparison against 20B-class editors.