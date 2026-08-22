# FLUX.2 Klein Weather Demo

All edits use transformer decomposition, `black-forest-labs/FLUX.2-klein-4B`, four inference steps, and an NVIDIA L4. Each case generates four seeded candidates; the verifier scores them against the Task 1 contract, hard-gates semantic drift (added people/vehicles, water turned to land, new water over solid ground), and keeps the best passing candidate. The selected candidate's pixels are returned untouched — no compositing or post-hoc tone mapping.

The edit matte licenses appearance change and defines leakage through its complement. The generation matte marks where new spatial texture is licensed. A per-candidate segmentation layout provides the drift gates. All are verification channels, not blend weights.

![Source, result, appearance matte, and generation matte](contact_sheet.jpg)

## Results

| Case | Target | Selected seed | Global MAE | Weak-support MAE | Structure edge F1 | Protected gain | Water loss | Water gain | Passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [urban_rain](results/urban_rain.png) | rain | 14 | 0.134 | 0.123 | 0.874 | 0.000 | 0.000 | 0.000 | yes |
| [mountain_fog](results/mountain_fog.png) | fog | 25 | 0.147 | 0.121 | 0.486 | 0.000 | 0.000 | 0.018 | yes |
| [coastal_snow](results/coastal_snow.png) | snow | 54 | 0.279 | 0.293 | 0.756 | 0.000 | 0.006 | 0.003 | yes |

Weak-support MAE is intentionally nonzero: weather legitimately changes illumination across the whole frame, so pixel identity outside the mattes is not a gate. Fog and diffuse rain attenuate edge contrast, so structure-edge thresholds are calibrated per target rather than compared directly across weather types.

## Observations

- **urban_rain**: The minimal per-target prompt passed all gates on every seed; the earlier constraint-list prompt invented foreground ponds on 4/4 seeds and one phantom protected instance, all caught by the water-gain and protected-gain gates. Masonry edges, the minaret, and the dry-scrub terrain are retained under a coherent storm sky.
- **mountain_fog**: Fog thickens with distance as a volume rather than a flat veil; the lake outline, shoreline, and foreground rock remain in place. One rejected seed had turned most of the lake into fogged-over terrain (water loss 0.61).
- **coastal_snow**: Snow keeps its explicit liquid-water prompt constraints — the minimal prompt froze the entire sea on every seed. The gated selection rejected candidates that extended shorelines or grew snow banks over open water (water loss 0.10–0.14 in the final batch, up to 0.48 across scanned seeds, versus 0.006 selected); accumulation lands on islands, rocks, and terrain while the sea stays liquid and lighthouse geometry is intact.

## Source attribution

- [20110102 Kharanaq old city Iran.jpg](https://commons.wikimedia.org/wiki/File:20110102_Kharanaq_old_city_Iran.jpg) by User:Ggia, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0).
- [1 lake louise pano 2019.jpg](https://commons.wikimedia.org/wiki/File:1_lake_louise_pano_2019.jpg) by Chensiyuan, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0).
- [2016-11-13 01 Kjeungskjær Lighthouse, Sør-Trøndelag, Norway.jpg](https://commons.wikimedia.org/wiki/File:2016-11-13_01_Kjeungskj%C3%A6r_Lighthouse,_S%C3%B8r-Tr%C3%B8ndelag,_Norway.jpg) by Gordon Leggett, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0).

All generated outputs are experimental synthetic weather visualizations, not records of actual conditions. Source photographs and their derivative edits remain subject to the corresponding CC BY-SA license listed above; code and documentation use the repository license.
