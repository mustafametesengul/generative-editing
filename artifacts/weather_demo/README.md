# FLUX.2 Klein Weather Demo

These edits use transformer decomposition, FLUX.2 Klein with four steps, and an NVIDIA L4. Each example has four candidates. The verifier rejects added protected content, lost water, and new water, then keeps the best result.

The edit matte shows where appearance may change. The generation matte shows where new weather detail should appear, but the prototype does not enforce it yet.

![Source, result, appearance matte, and generation matte](contact_sheet.jpg)

## Results

| Case | Target | Selected seed | Global MAE | Weak-support MAE | Structure edge F1 | Protected gain | Water loss | Water gain | Passed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [urban_rain](results/urban_rain.png) | rain | 14 | 0.134 | 0.123 | 0.874 | 0.000 | 0.000 | 0.000 | yes |
| [mountain_fog](results/mountain_fog.png) | fog | 25 | 0.147 | 0.121 | 0.486 | 0.000 | 0.000 | 0.018 | yes |
| [coastal_snow](results/coastal_snow.png) | snow | 54 | 0.279 | 0.293 | 0.756 | 0.000 | 0.006 | 0.003 | yes |

Weak-support MAE is not expected to be zero because weather changes light across the frame. Edge F1 helps rank candidates but is not a hard gate in this prototype.

## Observations

- **urban_rain:** all seeds passed. The selected edit keeps the masonry, minaret, dry terrain, and object count.
- **mountain_fog:** fog grows with distance while the lake and foreground stay in place. One seed was rejected for losing 61% of the water mask.
- **coastal_snow:** rejected seeds moved shorelines or covered open water. The selected edit has 0.6% water loss and keeps the lighthouse intact.

## Source attribution

- [20110102 Kharanaq old city Iran.jpg](https://commons.wikimedia.org/wiki/File:20110102_Kharanaq_old_city_Iran.jpg) by User:Ggia, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0).
- [1 lake louise pano 2019.jpg](https://commons.wikimedia.org/wiki/File:1_lake_louise_pano_2019.jpg) by Chensiyuan, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0).
- [2016-11-13 01 Kjeungskjær Lighthouse, Sør-Trøndelag, Norway.jpg](https://commons.wikimedia.org/wiki/File:2016-11-13_01_Kjeungskj%C3%A6r_Lighthouse,_S%C3%B8r-Tr%C3%B8ndelag,_Norway.jpg) by Gordon Leggett, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0).

These outputs are synthetic weather visualizations, not records of real conditions. Source photos and edits keep their listed CC BY-SA licenses; code and documentation use the repository license.
