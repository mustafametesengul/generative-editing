# Task 3: Fine-Tuning and Distillation Plan

Adapt the editor to weather, then make it faster without losing the source scene.

This is the production training plan. No training or distillation code is included in the prototype.

## Training data

Build about 50,000 licensed examples:

| Source | Share | Why it is useful |
| --- | ---: | --- |
| Same-place photos in different weather | 40% | Most realistic supervision |
| Teacher edits that pass gates and human review | 30% | Scales useful examples |
| Synthetic weather with exact masks | 30% | Gives precise labels |

Reject pairs where objects, geometry, or season changed. Keep at least half of each batch grounded in real photos. Hold out whole locations and cameras, with extra tests for night, reflections, text, people, and occlusion.

## Fine-tuning

Train a rank-32 LoRA on attention and MLP layers. Freeze the base transformer, VAE, and text encoder. Combine the normal flow loss with weather, edge, OCR, and identity losses. Include general editing data to reduce forgetting.

Run a 10,000-example pilot and scale only if evaluation improves. Add a conditioning adapter only if prompts are not enough. Full fine-tuning is the last resort.

## Distillation

Distill the four-step teacher into a two-step student. Start at 4B; try 2B only after that works. Match the teacher's denoising path and final image, and keep preservation losses so the student cannot ignore the source.

Then test INT8/FP8 quantization, keep sensitive VAE layers in FP16/BF16, and build a TensorRT engine. Judge the full editing task, not reconstruction error alone.

## Evaluation loop

Every build runs the same frozen suite: weather success, preservation, realism, human preference, p50/p95 latency, and peak VRAM. Report each weather, scene type, and hard case separately.

Release requires no drop in gate pass rate, at least 45% pairwise preference against the teacher, and at least 1.8x student speedup. Any text, identity, object-count, or safety regression blocks release. Version each model, engine, and dataset for simple rollback.

## Budget and risk

| Stage | Rough budget | Cadence |
| --- | ---: | --- |
| Data curation and probes | 2–4 GPU-weeks + labeling | Initial, then continuous |
| LoRA pilot | 4×A100 × 24 h | Per material change |
| Full LoRA run | 8×A100 × 48 h | Quarterly or drift-triggered |
| Two-step distillation | 8×A100/H100 × 72 h | After teacher promotion |
| TensorRT calibration + eval | 2–4 L4s × 1–2 days | Every release candidate |

**Main risk:** the two-step student may lose small source details before the image looks obviously worse. Distill only from a strong teacher, keep preservation losses, and test hard cases separately. If the student misses a gate, ship the four-step model. Training uses A100/H100 GPUs; inference stays on the L4.
