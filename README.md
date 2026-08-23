# Selective Weather Editing

This project changes weather in outdoor photos while keeping the camera, objects, people, text, and scene layout fixed.

The pipeline has three steps: **generate, verify, select**. FLUX.2 Klein creates several edits. A scene map checks each one for unwanted changes, and only the best passing result is returned.

| Before | After (snow) |
| --- | --- |
| ![Coastal source photograph](artifacts/weather_demo/sources/coastal.jpg) | ![Snow edit: accumulation on land and roofs while the sea stays liquid](artifacts/weather_demo/results/coastal_snow.png) |

See [the demo](artifacts/weather_demo/README.md) for more examples, masks, and metrics.

## What is included

- **Working prototype:** scene decomposition, FLUX or mock editing, candidate gates and ranking, metrics, and debug masks.
- **Production design:** OCR and identity checks, realism scoring, TensorRT, video consistency, provenance, watermarking, fine-tuning, and distillation.

The prototype demonstrates the data flow; it is not a production safety claim. Its semantic masks cannot prove unchanged identity or object count. The CLI writes no output when every candidate fails.

## Deliverables

- [Domain and assumptions](docs/ASSUMPTIONS.md)
- [Task 1: understanding the scene](docs/TASK1_PERCEPTION.md)
- [Task 2: editing and evaluation](docs/TASK2_GENERATION.md)
- [Task 3: training and deployment](docs/TASK3_TRAINING_DISTILLATION.md)
- [AI usage disclosure](docs/AI_USAGE.md)
- [Task 1 prototype](src/generative_editing/decomposition.py)
- Task 2: [editing](src/generative_editing/editing.py), [pipeline](src/generative_editing/pipeline.py), and [evaluation](src/generative_editing/evaluation.py)
- [L4 demo and attribution](artifacts/weather_demo/README.md)

## Setup and checks

```bash
uv sync
uv run pytest -q
```

## Run without model weights

The default path needs no model weights:

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

This downloads FLUX.2 Klein, SegFormer-B2, and Depth Anything V2 Small from Hugging Face. No weights are stored in this repository. Use `--cpu-offload` below 13 GiB, with a large speed penalty.

FLUX generation is capped near one megapixel. Larger inputs are downsampled before editing and resized back afterward, which preserves dimensions but can lose fine detail.

## Hardware recommendation

FLUX.2 Klein is documented at about 13 GiB VRAM, so the inference target is a 24 GiB L4. A100/H100 GPUs are reserved for training, distillation, and large evaluations.
