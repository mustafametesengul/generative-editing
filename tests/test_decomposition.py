import numpy as np
from PIL import Image, ImageDraw

from generative_editing.decomposition import HeuristicSceneDecomposer, Weather


def synthetic_scene() -> Image.Image:
    image = Image.new("RGB", (160, 120), "#75b9ed")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 60, 159, 119), fill="#648c45")
    draw.rectangle((55, 35, 105, 95), fill="#b94d3e")
    return image


def test_heuristic_decomposition_finds_sky_and_structure() -> None:
    decomposition = HeuristicSceneDecomposer().decompose(synthetic_scene())

    assert decomposition.sky.shape == (120, 160)
    assert decomposition.sky[10, 10] > 0.5
    assert decomposition.weather_surface[110, 10] > decomposition.weather_surface[20, 10]
    assert decomposition.structure_guard[60, 20] > 0.5


def test_edit_mattes_are_bounded_and_weather_specific() -> None:
    decomposition = HeuristicSceneDecomposer().decompose(synthetic_scene())
    rain = decomposition.edit_matte(Weather.RAIN)
    snow = decomposition.edit_matte(Weather.SNOW)

    assert np.isfinite(rain).all()
    assert 0.0 <= rain.min() <= rain.max() <= 1.0
    assert snow[110, 10] > rain[110, 10]