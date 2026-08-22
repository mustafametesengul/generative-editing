import numpy as np
from PIL import Image, ImageDraw

from generative_editing.decomposition import (
    WEATHER_SURFACE_NAMES,
    HeuristicSceneDecomposer,
    Weather,
    _build_structure_guard,
)


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
    rain_generation = decomposition.generation_matte(Weather.RAIN)
    snow = decomposition.edit_matte(Weather.SNOW)
    snow_generation = decomposition.generation_matte(Weather.SNOW)
    fog = decomposition.edit_matte(Weather.FOG)

    assert np.isfinite(rain).all()
    assert 0.0 <= rain.min() <= rain.max() <= 1.0
    assert snow_generation[110, 10] > rain_generation[110, 10]
    assert snow.min() >= 0.54
    assert snow_generation.min() < 0.20
    assert fog[20, 20] > fog[110, 20]


def test_structure_guard_ignores_transient_sky_texture() -> None:
    edges = np.zeros((32, 32), dtype=np.uint8)
    edges[8, 8:24] = 255
    edges[24, 8:24] = 255
    sky = np.zeros((32, 32), dtype=np.float32)
    sky[:16] = 1.0

    guard = _build_structure_guard(edges, sky)

    assert guard[8, 16] == 0.0
    assert guard[24, 16] == 1.0


def test_snow_surface_ontology_includes_mountain_land() -> None:
    assert {"hill", "mountain", "rock"} <= WEATHER_SURFACE_NAMES