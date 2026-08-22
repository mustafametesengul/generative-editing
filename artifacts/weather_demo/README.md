# FLUX.2 Klein Weather Demo

All edits use transformer decomposition, `black-forest-labs/FLUX.2-klein-4B`, four inference steps, and an NVIDIA L4.

The appearance matte controls geometry-preserving color and illumination transfer. The generation matte controls where raw generated spatial content may enter.

![Source, result, appearance matte, and generation matte](contact_sheet.jpg)

## Results

| Case | Target | Seed | Global MAE | Strong-support MAE | Weak-support MAE | Structure edge F1 |
|---|---:|---:|---:|---:|---:|---:|
| [urban_rain](results/urban_rain.png) | rain | 11 | 0.066 | 0.133 | 0.025 | 0.986 |
| [mountain_fog](results/mountain_fog.png) | fog | 23 | 0.114 | 0.130 | 0.069 | 0.638 |
| [coastal_snow](results/coastal_snow.png) | snow | 37 | 0.169 | 0.208 | 0.131 | 0.937 |

Fog intentionally attenuates edge contrast, so structure-edge thresholds are calibrated per target rather than compared directly across weather types.

## Observations

- **urban_rain**: Clouds and rain enter mainly through sky/atmosphere; ruins receive coherent darker illumination while static edges remain stable.
- **mountain_fog**: Fog attenuates distant terrain through the depth-conditioned generation matte while foreground geometry is retained.
- **coastal_snow**: Snow affects global illumination and receptive surfaces; the sea stays liquid and invented foreground geometry is rejected.

## Source attribution

- [20110102 Kharanaq old city Iran.jpg](https://commons.wikimedia.org/wiki/File:20110102_Kharanaq_old_city_Iran.jpg) by User:Ggia, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0).
- [1 lake louise pano 2019.jpg](https://commons.wikimedia.org/wiki/File:1_lake_louise_pano_2019.jpg) by Chensiyuan, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0).
- [2016-11-13 01 Kjeungskjær Lighthouse, Sør-Trøndelag, Norway.jpg](https://commons.wikimedia.org/wiki/File:2016-11-13_01_Kjeungskj%C3%A6r_Lighthouse,_S%C3%B8r-Tr%C3%B8ndelag,_Norway.jpg) by Gordon Leggett, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0).

All generated outputs are experimental synthetic weather visualizations, not records of actual conditions. Source photographs and their derivative edits remain subject to the corresponding CC BY-SA license listed above; code and documentation use the repository license.
