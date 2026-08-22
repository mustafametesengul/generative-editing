# Domain, Edit Contract, and Assumptions

## Domain choice

The domain is **single outdoor photos with the weather changed**: clear, overcast, rain, snow, or fog. Weather is a good testbed because it is neither one object nor a global style. It touches the sky, visibility over distance, lighting, falling particles, and how surfaces look (wet roads, snow cover). The system has to change all of that without moving an edge, changing a face, breaking text, or inventing an object. This makes preservation failures much easier to spot than they would be in a simple recoloring task.

The design carries over to other domains by swapping what counts as editable and how it is checked; the perception contract, editor, verifier, and gates stay the same.

## Edit contract

| Category | Attributes |
|---|---|
| Editable | Sky and clouds; precipitation; fog and visibility; overall lighting and color temperature; how surfaces respond (wetness, reflections, snow cover). |
| Hard invariant | Camera viewpoint and crop; object count, identity, pose, and position; geometry and silhouettes; faces; text, signs, logos, license plates. |
| Soft invariant | Materials, local texture, fine edges, and scene meaning. Their brightness and color may shift with the new lighting, but their structure must not. |
| Out of scope | Time-of-day or season changes; adding or removing objects; physically accurate weather simulation; forensic or evidentiary use. |

## Assumptions

1. Inputs are licensed or user-owned photos. They may incidentally contain people or license plates, so everything is treated as sensitive.
2. The main path handles one image up to about one megapixel. Bursts and video use the consistency extension in Task 2.
3. The request is one of five fixed weather targets, not free editing text. An intensity slider is future work.
4. There is no single correct output. Evaluation asks three questions: Did the weather change? Did everything else stay the same? Does it look real? Human reviewers also compare outputs side by side.
5. When in doubt, preserve. Snow hidden behind a readable sign beats plausible snow that breaks the sign.
6. Outputs are visualizations, not records of real conditions. Production outputs carry provenance metadata and a visible disclosure where needed.
7. Inference runs locally in a trusted environment; raw inputs never go to a third-party API.
8. The inference GPU is an NVIDIA L4 (24 GiB); the demo artifacts were generated on one.
9. Model choices are current as of 22 August 2026 and should be re-checked against the same benchmark when new checkpoints appear.
10. No model weights, sensitive samples, or claims about real events are included in this submission.