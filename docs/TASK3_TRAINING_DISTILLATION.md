# Task 3: Fine-Tuning and Distillation Plan

## Data

Build ~50k licensed tuples: 40% registered real before/after or burst pairs, 30% real images with teacher-generated candidates that passed the preservation gates and human review, 30% physically rendered synthetic pairs with exact masks. Keep ≥50% real per batch. Attach weather labels, decomposition channels, guards, and provenance. Hold out entire locations, sessions, and cameras; keep a small adjudicated “red” set (night, reflections, occlusion, ambiguous boundaries).

## Fine-tuning

Rank-32 LoRA on the rectified-flow transformer attention/MLP projections; VAE and text encoder frozen. A zero-initialized conditioning adapter on decomposition channels is added only if prompting plus verified selection underperforms. Mix generic edit examples to preserve base capability. Loss: native flow matching plus weather-classification, masked-LPIPS preservation, edge/OCR/identity, and leakage penalties. Run a 10k-example pilot first; promote only if it moves the edit-vs-preservation Pareto frontier on real validation data. Full fine-tuning is a last resort after LoRA saturates.

## Distillation

Use the accepted four-step model as teacher and train a two-step student (same 4B first, then a 2B width-reduced variant) with consistency/trajectory distillation, keeping the preservation losses so the student does not get fast by ignoring source detail. Then weight-only INT8/FP8 where the L4 supports it, VAE-sensitive layers at FP16/BF16, TensorRT compile, per-stratum calibration. Quantization is accepted on task metrics, not reconstruction error.

## Evaluation loop

Every candidate rebuild reruns the frozen suite: target-weather success, masked LPIPS/SSIM, edge/OCR/identity consistency, layout-drift gates, KID, human pairwise preference, p50/p95 latency, peak VRAM. Hard regressions in text, identity, object count, or safety block release. A shadow deployment feeds consented failures into the active-learning queue; retraining is monthly at first, then drift-triggered. Base model, LoRA, student, engine, and data snapshot are independently versioned for rollback.

## Budget and risk

| Stage | Rough budget | Cadence |
|---|---:|---|
| Data curation and probes | 2–4 GPU-weeks + labeling | Initial, then continuous |
| LoRA pilot | 4×A100 × 24 h | Per material change |
| Full LoRA run | 8×A100 × 48 h | Quarterly or drift-triggered |
| Two-step distillation | 8×A100/H100 × 72 h | After teacher promotion |
| TensorRT calibration + eval | 2–4 L4s × 1–2 days | Every release candidate |

The riskiest step is distillation: few-step students lose subtle source conditioning before aggregate realism drops. Mitigation: distill only from a strong teacher, keep explicit preservation losses, evaluate per hard stratum, and ship the four-step path if the two-step student misses any gate. Training runs on A100/H100; the L4 stays the inference target.