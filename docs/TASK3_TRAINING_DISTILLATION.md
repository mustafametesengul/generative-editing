# Task 3: Fine-Tuning and Distillation Plan

## Data

Build ~50k licensed training examples: 40% real photo pairs of the same scene in different weather, 30% real photos with teacher-model edits that passed the gates and a human check, 30% synthetic renders with exact masks. Keep at least half of every batch real. Hold out entire locations and cameras for testing, plus a hand-picked set of hard cases (night, reflections, occlusion, ambiguous boundaries).

## Fine-tuning

Rank-32 LoRA on the transformer's attention/MLP layers; the VAE and text encoder stay frozen. A conditioning adapter for the decomposition channels is added only if prompting plus selection stops being enough. Mix in generic editing examples so the model keeps its base skills. Loss: the native flow-matching loss plus penalties for wrong weather, damaged structure/text/identity, and leakage. Pilot on 10k examples first; scale up only if the pilot actually improves the edit-vs-preservation trade-off. Full fine-tuning is a last resort.

## Distillation

The fine-tuned four-step model becomes the teacher for a two-step student (same 4B size first, a smaller 2B later) trained with consistency distillation. Keep the preservation losses during distillation. Without them, the student can get faster by ignoring the source. Then use INT8/FP8 quantization where the L4 supports it (VAE-sensitive layers stay FP16/BF16), followed by a TensorRT build. Quantization is judged on task metrics, not reconstruction error.

## Evaluation loop

Every candidate build reruns the frozen suite: weather success, preservation metrics, drift gates, realism, human preference, latency, and VRAM. Regressions in text, identity, object count, or safety block the release. Failures from production feed the next training round; model, LoRA, student, engine, and data snapshot are versioned separately so rollback is one step.

## Budget and risk

| Stage | Rough budget | Cadence |
| --- | ---: | --- |
| Data curation and probes | 2–4 GPU-weeks + labeling | Initial, then continuous |
| LoRA pilot | 4×A100 × 24 h | Per material change |
| Full LoRA run | 8×A100 × 48 h | Quarterly or drift-triggered |
| Two-step distillation | 8×A100/H100 × 72 h | After teacher promotion |
| TensorRT calibration + eval | 2–4 L4s × 1–2 days | Every release candidate |

The riskiest step is distillation: few-step students tend to lose subtle source details before overall image quality visibly drops. Mitigations: distill only from a strong teacher, keep the preservation losses, evaluate the hard cases separately, and ship the four-step model if the two-step one misses any gate. Training runs on A100/H100; the L4 stays the inference target.
