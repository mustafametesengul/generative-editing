# Domain and Assumptions

## Domain choice

The domain is **changing weather in a single outdoor photo**: clear, overcast, rain, snow, or fog. Weather is a useful test because it changes sky, light, visibility, and surfaces at once. Yet the camera, objects, people, text, and geometry must stay fixed. That makes mistakes easy to see.

The same design can serve another domain by changing the edit rules and checks while keeping the map, editor, verifier, and gates.

## Edit contract

| Category | Attributes |
| --- | --- |
| May change | Sky, precipitation, fog, lighting, wetness, reflections, and snow cover |
| Must not change | Camera, crop, objects, identity, pose, geometry, faces, text, signs, logos, and plates |
| May shift slightly | Material color and brightness, but not texture or structure |
| Out of scope | Time or season changes, adding objects, physical simulation, or evidentiary use |

## Assumptions

1. Inputs are licensed or user-owned. Because they may show people or plates, treat all inputs as sensitive.
2. The prototype edits at about one megapixel. Larger photos are reduced and resized back, which can lose detail. Video is not implemented.
3. Users choose one of five weather targets. There is no free-text editing or intensity control.
4. Evaluation asks: Did the weather change? Is it still the same scene? Does it look real? Human reviewers compare images side by side.
5. Preservation wins when goals conflict. A readable sign matters more than perfect snow.
6. Outputs are synthetic visualizations, not evidence of real conditions. Production needs provenance and a visible disclosure; the prototype does not add them.
7. Inference is local. The target is an NVIDIA L4 (24 GiB), but peak memory and latency must be measured end to end.
8. Model choices are current as of 22 August 2026. This repository includes no model weights, sensitive samples, or claims about real events.
