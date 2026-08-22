from PIL import Image, ImageDraw

from generative_editing.decomposition import HeuristicSceneDecomposer, Weather
from generative_editing.editing import MockWeatherEditor
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
    assert result.metrics.inside_edit_mae > result.metrics.outside_mae
    assert result.metrics.outside_psnr > 25.0
    assert (tmp_path / "edit_matte.png").is_file()