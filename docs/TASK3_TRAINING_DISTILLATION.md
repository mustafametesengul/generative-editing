# Task 3: Fine-Tuning and Distillation Plan

## Objective and data

Fine-tune FLUX.2 Klein 4B for weather edits that improve target attainment without increasing protected-region drift, then compress the path from its current four steps to a two-step student if quality gates permit. Build approximately 50,000 licensed training tuples: 40% registered real before/after or burst pairs, 30% single real images with teacher-generated candidates accepted by preservation probes and human review, and 30% physically rendered synthetic pairs with depth and exact masks. Sample batches to keep at least half their images real. Attach source/target weather, decomposition channels, text/OCR/identity guards, uncertainty, and provenance. Hold out entire locations, capture sessions, cameras, and contributors; maintain a small manually adjudicated “red” set for ambiguous boundaries, night, reflections, occlusion, and extreme weather.

## Fine-tuning

Start with rank-32 LoRA on the rectified-flow transformer attention and selected MLP projections; freeze the VAE and text encoder. Train a small zero-initialized conditioning adapter for the sky, far-depth, surface, and guard channels only if instruction tuning plus compositing underperforms. This minimizes catastrophic drift and keeps deployment modular. Mix ordinary generation/edit examples to preserve base capability. Optimize the native flow-matching objective plus weather classification, masked perceptual preservation, edge/OCR/identity consistency, and leakage penalties:

$$
\mathcal{L}=\mathcal{L}_{flow}+\lambda_e\mathcal{L}_{weather}+\lambda_p\mathcal{L}_{masked\text{-}LPIPS}+\lambda_g\mathcal{L}_{edge/OCR}+\lambda_l\mathcal{L}_{leakage}.
$$

Run a 10k-example pilot first. Promote to the full run only if it moves the edit-versus-preservation Pareto frontier on real validation data. Full fine-tuning is a fallback only when LoRA saturates and ablations show broad transformer adaptation is required.

## Distillation and deployment

Use the accepted four-step domain model as teacher. Record teacher velocity/trajectory targets at paired noise times, then train a smaller student (initially the same 4B architecture for risk reduction, later a 2B width-reduced transformer) with consistency/trajectory distillation so two-step predictions agree along the teacher’s probability-flow path. Retain the domain preservation losses during distillation; otherwise the student may become fast by ignoring source detail. Compare two-step, three-step, and four-step variants before choosing. Apply weight-only INT8 or FP8 where the L4 engine supports it, keep VAE numerically sensitive layers at FP16/BF16, compile with TensorRT, and calibrate on every weather/scene stratum. Quantization is accepted only after task metrics, not from reconstruction error alone.

The release loop reruns target-weather success, masked LPIPS/SSIM, edge/object/OCR/identity consistency, KID, human pairwise preference, p50/p95 latency, throughput, and peak VRAM. Hard regressions in text, identity, object count, safety, or provenance block release. A shadow deployment samples consented failures into the active-learning queue; labels and retraining are monthly at first, then drift-triggered. Base model, LoRA, student, engine, data snapshot, and thresholds are independently versioned for rollback.

## Budget, cadence, and risk

| Stage | Rough budget | Cadence |
|---|---:|---|
| Data curation, teacher proposals, probes | 2-4 L4/A100 weeks plus labeling | Initial build, then continuous sampling |
| LoRA pilot | 4 x A100-80GB for 24 h (96 GPU-h) | Per material model/data change |
| Full LoRA/adapter run | 8 x A100-80GB for 48 h (384 GPU-h) | Quarterly or drift-triggered |
| Two-step consistency distillation | 8 x A100/H100 for 72 h (576 GPU-h) | Infrequent, after teacher promotion |
| TensorRT calibration and evaluation | 2-4 L4s for 1-2 days | Every release candidate |

These are planning estimates to refine after the pilot. The riskiest step is distillation: few-step students often lose subtle source conditioning before aggregate realism changes. The mitigation is to distill only after a strong teacher, retain explicit preservation losses, evaluate by hard stratum, and ship the four-step L4 path if the two-step model does not strictly satisfy gates. Switch to an A100/H100 runtime for this training work; the L4 remains sufficient for inference and smoke evaluation.