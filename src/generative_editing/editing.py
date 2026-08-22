"""Image editors and selective compositing for weather transformations."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from generative_editing.decomposition import Weather


MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


class ImageEditor(Protocol):
    def edit(self, image: Image.Image, weather: Weather, seed: int) -> Image.Image: ...


class Flux2KleinEditor:
    """Lazy Diffusers adapter for the open-weight FLUX.2 Klein 4B model."""

    def __init__(self, model_id: str = MODEL_ID, cpu_offload: bool = False) -> None:
        import torch
        from diffusers import Flux2KleinPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("FLUX.2 inference requires a CUDA device; use --editor mock without one")
        total_vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_vram_gib < 13.0 and not cpu_offload:
            raise RuntimeError(
                f"FLUX.2 Klein needs about 13 GiB VRAM; found {total_vram_gib:.1f} GiB. "
                "Enable CPU offload or use a larger GPU."
            )

        self._torch = torch
        self.pipeline = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        if cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to("cuda")

    def edit(self, image: Image.Image, weather: Weather, seed: int) -> Image.Image:
        width, height = _generation_size(image.width, image.height)
        generator = self._torch.Generator(device="cuda").manual_seed(seed)
        result = self.pipeline(
            image=image.convert("RGB"),
            prompt=weather_prompt(weather),
            height=height,
            width=width,
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=generator,
        ).images[0]
        return result.resize(image.size, Image.Resampling.LANCZOS)


class MockWeatherEditor:
    """Deterministic, weight-free editor that exercises the complete data flow."""

    def edit(self, image: Image.Image, weather: Weather, seed: int) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        result = rgb.astype(np.float32)
        random = np.random.default_rng(seed)

        if weather is Weather.CLEAR:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= 1.15
            hsv[..., 2] *= 1.08
            result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        elif weather is Weather.OVERCAST:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None]
            result = 0.72 * result + 0.28 * gray
            result *= 0.88
        elif weather is Weather.RAIN:
            result *= np.array([0.72, 0.78, 0.86], dtype=np.float32)
            result = _draw_rain(result, random)
        elif weather is Weather.SNOW:
            result = result * np.array([0.82, 0.87, 0.94], dtype=np.float32) + 28.0
            result = _draw_snow(result, random)
        elif weather is Weather.FOG:
            height = rgb.shape[0]
            veil = np.linspace(0.62, 0.25, height, dtype=np.float32)[:, None, None]
            result = result * (1.0 - veil) + 225.0 * veil

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def selective_composite(original: Image.Image, candidate: Image.Image, matte: np.ndarray) -> Image.Image:
    """Blend only decomposed weather regions back into the source photograph."""
    source = np.asarray(original.convert("RGB"), dtype=np.float32)
    edited = np.asarray(candidate.convert("RGB").resize(original.size), dtype=np.float32)
    if matte.shape != source.shape[:2]:
        raise ValueError("Edit matte dimensions must match the source image")
    alpha = np.clip(matte, 0.0, 1.0)[..., None].astype(np.float32)
    composite = source * (1.0 - alpha) + edited * alpha
    return Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8), mode="RGB")


def weather_prompt(weather: Weather) -> str:
    descriptions = {
        Weather.CLEAR: "clear blue-sky weather with crisp visibility and natural sunlight",
        Weather.OVERCAST: "realistic overcast weather with a continuous cloud layer and soft diffuse light",
        Weather.RAIN: "realistic rainy weather with wet ground, rain streaks, cloud cover, and coherent reflections",
        Weather.SNOW: "realistic snowy weather with falling snow and physically plausible snow accumulation",
        Weather.FOG: "realistic fog with depth-dependent atmospheric scattering and reduced distant visibility",
    }
    return (
        f"Change only the weather to {descriptions[weather]}. Preserve the exact camera viewpoint, crop, "
        "scene geometry, object count and placement, people and object identities, text, logos, architecture, "
        "and fine edges. Do not add or remove objects. Keep this a photorealistic edit of the same photograph."
    )


def _generation_size(width: int, height: int, max_pixels: int = 1024 * 1024) -> tuple[int, int]:
    scale = min(1.0, (max_pixels / (width * height)) ** 0.5)
    resized_width = max(64, round(width * scale / 16) * 16)
    resized_height = max(64, round(height * scale / 16) * 16)
    return resized_width, resized_height


def _draw_rain(image: np.ndarray, random: np.random.Generator) -> np.ndarray:
    canvas = np.clip(image, 0, 255).astype(np.uint8)
    height, width = canvas.shape[:2]
    overlay = canvas.copy()
    for _ in range(max(40, width * height // 1800)):
        x = int(random.integers(-10, width))
        y = int(random.integers(0, height))
        length = int(random.integers(7, 18))
        cv2.line(overlay, (x, y), (x + 3, min(height - 1, y + length)), (190, 205, 220), 1)
    return cv2.addWeighted(canvas, 0.82, overlay, 0.18, 0).astype(np.float32)


def _draw_snow(image: np.ndarray, random: np.random.Generator) -> np.ndarray:
    canvas = np.clip(image, 0, 255).astype(np.uint8)
    height, width = canvas.shape[:2]
    for _ in range(max(30, width * height // 2200)):
        center = (int(random.integers(0, width)), int(random.integers(0, height)))
        radius = int(random.integers(1, 3))
        cv2.circle(canvas, center, radius, (242, 246, 250), -1, lineType=cv2.LINE_AA)
    return canvas.astype(np.float32)