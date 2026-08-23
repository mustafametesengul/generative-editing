# Task 3: Fine-Tuning and Distillation Plan

## Data

Build ~50k licensed training examples: 40% registered real photo pairs of the same scene in different weather, 30% real source photos with teacher edits that passed the gates and human review, and 30% synthetic renders with exact masks. Reject real pairs with geometry, season, or object changes after registration; otherwise the model learns to violate invariants. Keep at least half of each batch grounded in real source images. Hold out entire locations and cameras for testing, plus hard slices for night, reflections, occlusion, text, people, and ambiguous boundaries.

## Fine-tuning

Train rank-32 LoRA weights on the transformer's attention/MLP layers; freeze the VAE, text encoder, and base transformer weights. Add a separately trained conditioning adapter only if prompting plus selection misses the target. Alongside native flow-matching loss, decode a predicted clean image on a subset of batches and apply target-weather classifier loss, masked perceptual/edge loss outside licensed regions, and OCR/identity consistency loss where applicable. Mix in generic editing examples to limit catastrophic forgetting. Pilot on 10k examples and scale only if the frozen evaluation improves; full fine-tuning is a last resort.

## Distillation

The fine-tuned four-step model becomes the teacher for a two-step student (same 4B size first, a smaller 2B later) trained with consistency distillation on the same source/prompt pairs. Match teacher trajectories and clean-image predictions while retaining the preservation losses; otherwise the student can get faster by ignoring the source. Then test INT8/FP8 quantization where supported, leaving sensitive VAE layers in FP16/BF16, followed by a TensorRT build. Quantization is judged on end-to-end task metrics, not reconstruction error alone.

## Evaluation loop

Every candidate build reruns the frozen suite: weather success, preservation metrics, drift gates, realism, human preference, p50/p95 latency, and peak VRAM. Before training, set per-slice promotion bounds (for example, no hard-gate pass-rate drop, at least 45% pairwise preference against the teacher, and at least 1.8x measured speedup for the two-step student). Text, identity, object-count, or safety regressions block release regardless of average quality. Production failures feed the next training round; model, LoRA, student, engine, and data snapshot are versioned separately for one-step rollback.

## Budget and risk

| Stage | Rough budget | Cadence |
| --- | ---: | --- |
| Data curation and probes | 2–4 GPU-weeks + labeling | Initial, then continuous |
| LoRA pilot | 4×A100 × 24 h | Per material change |
| Full LoRA run | 8×A100 × 48 h | Quarterly or drift-triggered |
| Two-step distillation | 8×A100/H100 × 72 h | After teacher promotion |
| TensorRT calibration + eval | 2–4 L4s × 1–2 days | Every release candidate |

The riskiest step is distillation: few-step students tend to lose subtle source details before overall image quality visibly drops. Mitigations: distill only from a strong teacher, keep the preservation losses, evaluate the hard cases separately, and ship the four-step model if the two-step one misses any gate. Training runs on A100/H100; the L4 stays the inference target.
