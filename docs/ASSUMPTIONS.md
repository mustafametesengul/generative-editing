# Domain, Edit Contract, and Assumptions

## Domain choice

The domain is **single outdoor RGB photographs with weather changed among clear, overcast, rain, snow, and fog**. It is a useful testbed because weather is neither a single object nor a global style. It couples sky appearance, depth-dependent atmospheric transmission, illumination, transient particles, and material response such as wet roads or snow accumulation. A successful system must therefore make meaningful local and global changes without moving a building edge, changing a face, corrupting text, or inventing a car. Those tensions expose attribute entanglement and preservation failures better than a simple recoloring task.

The design generalizes to other edits by replacing the editable-layer ontology and domain probes while retaining the perception contract, conditioned editor, candidate verifier, and evaluation gates.

## Edit contract

| Category | Attributes |
|---|---|
| Editable | Sky/cloud state; precipitation; depth-dependent fog/visibility; diffuse illumination and color temperature; weather response on receptive surfaces (wetness, puddle-like reflection, snow cover). |
| Hard invariant | Camera viewpoint and crop; object count, identity, pose, and location; geometry and silhouettes; faces; text, signs, logos, and license plates. |
| Soft invariant | Base material identity and albedo, local texture, fine edges, and scene semantics. Their observed intensity may change under new illumination or occlusion, but structure must not. |
| Out of scope | Time-of-day or season change; adding/removing objects; physically certified weather simulation; meteorological reconstruction; forensic or evidentiary use. |

## Assumptions

1. Inputs are licensed or user-owned photographs. A frame may incidentally contain people or vehicle identifiers, so it is treated as sensitive during processing.
2. The main path handles one image up to about one megapixel. Bursts and video use the consistency extension described in Task 2.
3. The requested target is one of five controlled labels, not unrestricted editing text. Edit intensity is a future scalar control.
4. There is no unique ground-truth output. Evaluation uses target attainment, invariance, realism, and human pairwise judgments rather than pixel matching to one target.
5. Ambiguous conflicts resolve in favor of preservation. For example, snow behind a readable sign is preferred over plausible accumulation that corrupts the sign.
6. The output is a visualization, not evidence of actual conditions. Production outputs carry provenance metadata and a visible disclosure where context requires it.
7. Inference runs in a trusted environment. Raw inputs are not sent to a third-party model API in the proposed production design.
8. The expected inference GPU is an NVIDIA L4 with 24 GiB VRAM; the included demo artifacts were generated on one.
9. Model selection is current as of 22 August 2026 and must be revisited against the same evaluation set when newer checkpoints appear.
10. No model weights, real sensitive samples, or generated claims about real events are included in this submission.