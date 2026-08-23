import numpy as np
from PIL import Image, ImageDraw

from generative_editing.decomposition import (
    HeuristicSceneDecomposer,
    SceneLayout,
    Weather,
)
from generative_editing.editing import MockWeatherEditor, weather_prompt
from generative_editing.evaluation import (
    EditMetrics,
    candidate_score,
    evaluate_edit,
    passes_gates,
)
from generative_editing.pipeline import WeatherEditingPipeline


def _scene() -> Image.Image:
    source = Image.new("RGB", (160, 120), "#75b9ed")
    draw = ImageDraw.Draw(source)
    draw.rectangle((0, 60, 159, 119), fill="#648c45")
    draw.rectangle((55, 35, 105, 95), fill="#b94d3e")
    return source


def test_pipeline_edits_weather_and_preserves_outside_region(tmp_path) -> None:
    source = _scene()
    pipeline = WeatherEditingPipeline(HeuristicSceneDecomposer(), MockWeatherEditor())

    result = pipeline.run(source, Weather.SNOW, seed=7)
    result.save_debug_masks(tmp_path)

    assert result.image.size == source.size
    source_array = np.asarray(source, dtype=np.float32) / 255.0
    result_array = np.asarray(result.image, dtype=np.float32) / 255.0
    assert np.abs(result_array - source_array).mean() > 0.02
    assert result.matte.min() >= 0.54
    assert (tmp_path / "edit_matte.png").is_file()


def test_pipeline_returns_generator_pixels_untouched() -> None:
    source = _scene()
    pipeline = WeatherEditingPipeline(HeuristicSceneDecomposer(), MockWeatherEditor())

    result = pipeline.run(source, Weather.RAIN, seed=3)

    expected = MockWeatherEditor().edit(source, Weather.RAIN, seed=3)
    assert result.seed == 3
    assert np.array_equal(np.asarray(result.image), np.asarray(expected))


class _SeedKeyedEditor:
    """Seed 0 destroys structure; seed 1 applies a gentle global tint."""

    def edit(self, image: Image.Image, weather: Weather, seed: int) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        if seed % 2 == 0:
            noise = np.random.default_rng(seed).uniform(0, 255, rgb.shape)
            return Image.fromarray(noise.astype(np.uint8), mode="RGB")
        return Image.fromarray(
            np.clip(rgb * 0.8 + 20.0, 0, 255).astype(np.uint8), mode="RGB"
        )


def test_pipeline_selects_more_preserving_candidate() -> None:
    source = _scene()
    pipeline = WeatherEditingPipeline(
        HeuristicSceneDecomposer(), _SeedKeyedEditor(), candidates=2
    )

    result = pipeline.run(source, Weather.OVERCAST, seed=0)

    assert result.seed == 1


def test_candidate_score_rejects_missing_edit() -> None:
    unedited = EditMetrics(
        global_edit_mae=0.001,
        semantic_support_mae=0.001,
        weak_support_mae=0.001,
        structure_edge_f1=1.0,
        protected_gain=0.0,
        water_loss=0.0,
        water_gain=0.0,
    )
    edited = EditMetrics(
        global_edit_mae=0.08,
        semantic_support_mae=0.10,
        weak_support_mae=0.03,
        structure_edge_f1=0.9,
        protected_gain=0.0,
        water_loss=0.0,
        water_gain=0.0,
    )

    assert candidate_score(unedited) == float("-inf")
    assert candidate_score(edited) > candidate_score(unedited)
    assert not passes_gates(unedited)
    assert passes_gates(edited)


def test_layout_drift_flags_added_people_and_land() -> None:
    shape = (100, 100)
    water = np.zeros(shape, dtype=bool)
    water[60:, :] = True
    empty = np.zeros(shape, dtype=bool)
    source_layout = SceneLayout(protected=empty, water=water, sky=empty)

    added_person = empty.copy()
    added_person[40:52, 45:50] = True
    shrunk_water = water.copy()
    shrunk_water[60:80, 30:70] = False
    drifted_layout = SceneLayout(protected=added_person, water=shrunk_water, sky=empty)

    new_pond = water.copy()
    new_pond[20:50, 10:60] = True
    ponded_layout = SceneLayout(protected=empty, water=new_pond, sky=empty)

    image = Image.new("RGB", shape[::-1], "gray")
    edited = Image.new("RGB", shape[::-1], "darkgray")
    matte = np.ones(shape, dtype=np.float32)

    clean = evaluate_edit(
        image,
        edited,
        matte,
        source_layout=source_layout,
        candidate_layout=source_layout,
    )
    drifted = evaluate_edit(
        image,
        edited,
        matte,
        source_layout=source_layout,
        candidate_layout=drifted_layout,
    )
    ponded = evaluate_edit(
        image,
        edited,
        matte,
        source_layout=source_layout,
        candidate_layout=ponded_layout,
    )

    assert (
        clean.protected_gain == 0.0
        and clean.water_loss == 0.0
        and clean.water_gain == 0.0
    )
    assert drifted.protected_gain > 0.002
    assert drifted.water_loss > 0.08
    assert ponded.water_gain > 0.02
    assert passes_gates(clean)
    assert not passes_gates(drifted)
    assert not passes_gates(ponded)
    assert candidate_score(drifted) < candidate_score(clean)
    assert candidate_score(ponded) < candidate_score(clean)


class _DriftKeyedEditor:
    """Every seed darkens; the decomposer stub reports drift for even seeds."""

    def edit(self, image: Image.Image, weather: Weather, seed: int) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        shade = 0.8 if seed % 2 else 0.79
        result = Image.fromarray(
            np.clip(rgb * shade + 15.0, 0, 255).astype(np.uint8), mode="RGB"
        )
        result.info["seed"] = seed
        return result


class _DriftKeyedDecomposer(HeuristicSceneDecomposer):
    def layout(self, image: Image.Image) -> SceneLayout:
        layout = super().layout(image)
        if image.info.get("seed", 1) % 2 == 0:
            protected = layout.protected.copy()
            protected[:20, :20] = True
            return SceneLayout(protected=protected, water=layout.water, sky=layout.sky)
        return layout


def test_pipeline_prefers_gated_candidate_over_higher_score() -> None:
    pipeline = WeatherEditingPipeline(
        _DriftKeyedDecomposer(), _DriftKeyedEditor(), candidates=2
    )

    result = pipeline.run(_scene(), Weather.OVERCAST, seed=0)

    assert result.seed == 1
    assert result.passed


class _NoEditEditor:
    def edit(self, image: Image.Image, weather: Weather, seed: int) -> Image.Image:
        return image.copy()


def test_pipeline_marks_result_failed_when_all_candidates_miss_edit_gate() -> None:
    pipeline = WeatherEditingPipeline(
        HeuristicSceneDecomposer(), _NoEditEditor(), candidates=2
    )

    result = pipeline.run(_scene(), Weather.OVERCAST, seed=0)

    assert not result.passed


def test_snow_prompt_preserves_water_and_object_inventory() -> None:
    prompt = weather_prompt(Weather.SNOW)

    assert "remain liquid and unfrozen" in prompt
    assert "Do not add or remove people" in prompt
    assert "land-water" in prompt


def test_non_snow_prompts_are_minimal() -> None:
    for weather in (Weather.CLEAR, Weather.OVERCAST, Weather.RAIN, Weather.FOG):
        prompt = weather_prompt(weather)
        assert prompt.startswith("Keep everything the same")
        assert "Do not" not in prompt
