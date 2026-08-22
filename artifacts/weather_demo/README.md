# FLUX.2 Klein Weather Demo

All edits use transformer decomposition, `black-forest-labs/FLUX.2-klein-4B`, four inference steps, and an NVIDIA L4.

![Source, result, and matte comparison](contact_sheet.jpg)

## Results

| Case | Target | Seed | Inside MAE | Outside MAE | Outside PSNR | Edge F1 |
|---|---:|---:|---:|---:|---:|---:|
| [urban_rain](results/urban_rain.png) | rain | 11 | 0.143 | 0.019 | 26.33 dB | 0.950 |
| [mountain_fog](results/mountain_fog.png) | fog | 23 | 0.114 | 0.068 | 20.24 dB | 0.958 |
| [coastal_snow](results/coastal_snow.png) | snow | 37 | 0.178 | 0.044 | 22.73 dB | 0.920 |

## Observations

- **urban_rain**: Convincing cloud and rain replacement; minaret and ruin geometry remain aligned.
- **mountain_fog**: Depth-aware distant attenuation is visible; residual bright cloud contours remain a refinement target.
- **coastal_snow**: Lighthouse is preserved, but surface snow is weak and bird-like sky artifacts violate object-count invariance.

## Source attribution

- [20110102 Kharanaq old city Iran.jpg](https://commons.wikimedia.org/wiki/File:20110102_Kharanaq_old_city_Iran.jpg) by User:Ggia, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0).
- [1 lake louise pano 2019.jpg](https://commons.wikimedia.org/wiki/File:1_lake_louise_pano_2019.jpg) by Chensiyuan, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0).
- [2016-11-13 01 Kjeungskjær Lighthouse, Sør-Trøndelag, Norway.jpg](https://commons.wikimedia.org/wiki/File:2016-11-13_01_Kjeungskj%C3%A6r_Lighthouse,_S%C3%B8r-Tr%C3%B8ndelag,_Norway.jpg) by Gordon Leggett, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0).

All generated outputs are experimental synthetic weather visualizations, not records of actual conditions. Source photographs and their derivative edits remain subject to the corresponding CC BY-SA license listed above; code and documentation use the repository license.
