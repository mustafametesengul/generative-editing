import numpy as np
from PIL import Image, ImageDraw

from generative_editing.decomposition import HeuristicSceneDecomposer, Weather
from generative_editing.editing import MockWeatherEditor, selective_composite, weather_prompt
from generative_editing.pipeline import WeatherEditingPipeline


def test_pipeline_edits_weather_and_preserves_outside_region(tmp_path) -> None:
    source = Image.new("RGB", (160, 120), "#75b9ed")
    draw = ImageDraw.Draw(source)
    draw.rectangle((0, 60, 159, 119), fill="#648c45")
    draw.rectangle((55, 35, 105, 95), fill="#b94d3e")
    pipeline = WeatherEditingPipeline(HeuristicSceneDecomposer(), MockWeatherEditor())

    result = pipeline.run(source, Weather.SNOW, seed=7)
    result.save_debug_masks(tmp_path)

    assert result.image.size == source.size
    source_array = np.asarray(source, dtype=np.float32) / 255.0
    result_array = np.asarray(result.image, dtype=np.float32) / 255.0
    assert np.abs(result_array - source_array).mean() > 0.02
    assert result.matte.min() >= 0.54
    assert (tmp_path / "edit_matte.png").is_file()


def test_compositor_applies_global_color_but_restores_guarded_detail() -> None:
    source_array = np.full((64, 64, 3), 50, dtype=np.uint8)
    source_array[:, 31:33] = 220
    candidate_array = np.full((64, 64, 3), (100, 130, 170), dtype=np.uint8)
    matte = np.ones((64, 64), dtype=np.float32)
    guard = np.zeros((64, 64), dtype=np.float32)
    guard[:, 29:35] = 1.0

    result = np.asarray(
        selective_composite(
            Image.fromarray(source_array),
            Image.fromarray(candidate_array),
            matte,
            guard,
        )
    )

    assert np.allclose(result[20, 5], candidate_array[20, 5], atol=1)
    assert result[20, 31].mean() > result[20, 28].mean() + 50


def test_compositor_rejects_generated_geometry_outside_receptive_regions() -> None:
    source_array = np.full((64, 64, 3), 80, dtype=np.uint8)
    candidate_array = np.full((64, 64, 3), 130, dtype=np.uint8)
    candidate_array[20:44, 20:44] = 10
    appearance_matte = np.ones((64, 64), dtype=np.float32)
    generation_matte = np.zeros((64, 64), dtype=np.float32)

    result = np.asarray(
        selective_composite(
            Image.fromarray(source_array),
            Image.fromarray(candidate_array),
            appearance_matte,
            generation_matte=generation_matte,
        )
    )

    assert result.mean() > source_array.mean()
    assert np.abs(result[32, 32].astype(int) - result[5, 5].astype(int)).max() <= 1


def test_snow_prompt_preserves_water_and_object_inventory() -> None:
    prompt = weather_prompt(Weather.SNOW)

    assert "remain liquid and unfrozen" in prompt
    assert "Do not add or remove people" in prompt


def test_fog_attenuates_more_source_detail_than_snow() -> None:
    source_array = np.full((64, 64, 3), 50, dtype=np.uint8)
    source_array[:, 31:33] = 220
    candidate_array = np.full((64, 64, 3), 170, dtype=np.uint8)
    matte = np.ones((64, 64), dtype=np.float32)
    guard = np.ones((64, 64), dtype=np.float32)

    snow_like = np.asarray(
        selective_composite(Image.fromarray(source_array), Image.fromarray(candidate_array), matte, guard)
    )
    fog_like = np.asarray(
        selective_composite(
            Image.fromarray(source_array),
            Image.fromarray(candidate_array),
            matte,
            guard,
            detail_strength=0.20,
        )
    )

    snow_contrast = snow_like[:, 31:33].mean() - snow_like[:, 25:27].mean()
    fog_contrast = fog_like[:, 31:33].mean() - fog_like[:, 25:27].mean()
    assert fog_contrast < snow_contrast


def test_rain_removes_warm_direct_sun_appearance_globally() -> None:
    source_array = np.empty((64, 64, 3), dtype=np.uint8)
    source_array[:, :32] = (210, 160, 90)
    source_array[:, 32:] = (105, 80, 45)
    candidate_array = np.empty((64, 64, 3), dtype=np.uint8)
    candidate_array[:, :32] = (150, 160, 170)
    candidate_array[:, 32:] = (75, 80, 85)
    appearance_matte = np.full((64, 64), 0.75, dtype=np.float32)
    generation_matte = np.zeros((64, 64), dtype=np.float32)

    result = np.asarray(
        selective_composite(
            Image.fromarray(source_array),
            Image.fromarray(candidate_array),
            appearance_matte,
            generation_matte=generation_matte,
            weather=Weather.RAIN,
        ),
        dtype=np.float32,
    )

    source_luminance = source_array.mean(axis=2)
    result_luminance = result.mean(axis=2)
    source_ratio = source_luminance[:, :32].mean() / source_luminance[:, 32:].mean()
    result_ratio = result_luminance[:, :32].mean() / result_luminance[:, 32:].mean()
    assert result.mean() < source_array.mean() - 15.0
    assert result_ratio < source_ratio - 0.35
    assert (result[..., 0] - result[..., 2]).mean() < 60.0