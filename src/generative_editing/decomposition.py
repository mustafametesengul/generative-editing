"""Weather-aware scene decomposition for selective image editing."""

from enum import StrEnum
from typing import Protocol, Self

import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, model_validator
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    AutoModelForSemanticSegmentation,
)

WEATHER_SURFACE_NAMES = {
    "earth",
    "field",
    "grass",
    "hill",
    "land",
    "mountain",
    "path",
    "road",
    "rock",
    "roof",
    "runway",
    "sand",
    "sidewalk",
    "stone",
}

WATER_NAMES = {"lake", "river", "sea", "swimming pool", "water", "waterfall"}

PROTECTED_NAMES = {
    "airplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "minibike",
    "person",
    "poster",
    "ship",
    "signboard",
    "trade name",
    "traffic light",
    "truck",
    "van",
}


class Weather(StrEnum):
    CLEAR = "clear"
    OVERCAST = "overcast"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"


class SceneDecomposition(BaseModel):
    """Soft masks describing editable weather layers and guarded structure."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    sky: np.ndarray
    atmosphere: np.ndarray
    weather_surface: np.ndarray
    structure_guard: np.ndarray
    confidence: np.ndarray

    @model_validator(mode="after")
    def _check_masks(self) -> Self:
        shape = self.sky.shape
        arrays = (
            self.sky,
            self.atmosphere,
            self.weather_surface,
            self.structure_guard,
            self.confidence,
        )
        if len(shape) != 2 or any(array.shape != shape for array in arrays):
            raise ValueError("All decomposition masks must have the same HxW shape")
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("Decomposition masks must contain finite values")
        if any(array.min() < 0.0 or array.max() > 1.0 for array in arrays):
            raise ValueError("Decomposition masks must be in the [0, 1] range")
        return self

    def edit_matte(self, weather: Weather) -> np.ndarray:
        """Build a soft support map for target-specific weather appearance."""
        weights = {
            Weather.CLEAR: (1.00, 0.20, 0.25, 0.15),
            Weather.OVERCAST: (1.00, 0.30, 0.35, 0.65),
            Weather.RAIN: (0.90, 0.65, 0.50, 0.75),
            Weather.SNOW: (0.95, 0.85, 0.55, 0.55),
            Weather.FOG: (0.60, 0.20, 1.00, 0.20),
        }
        sky_weight, surface_weight, atmosphere_weight, global_weight = weights[weather]
        semantic_support = np.maximum.reduce(
            (
                sky_weight * self.sky,
                surface_weight * self.weather_surface,
                atmosphere_weight * self.atmosphere,
            )
        )
        semantic_support *= 0.25 + 0.75 * self.confidence
        matte = global_weight + (1.0 - global_weight) * semantic_support
        matte = cv2.GaussianBlur(matte.astype(np.float32), (0, 0), sigmaX=1.2)
        return np.clip(matte, 0.0, 1.0)

    def generation_matte(self, weather: Weather) -> np.ndarray:
        """Limit generated spatial content to physically receptive regions."""
        weights = {
            Weather.CLEAR: (1.00, 0.10, 0.05),
            Weather.OVERCAST: (1.00, 0.15, 0.10),
            Weather.RAIN: (0.90, 0.75, 0.30),
            Weather.SNOW: (0.95, 0.90, 0.15),
            Weather.FOG: (0.60, 0.15, 1.00),
        }
        sky_weight, surface_weight, atmosphere_weight = weights[weather]
        matte = np.maximum.reduce(
            (
                sky_weight * self.sky,
                surface_weight * self.weather_surface,
                atmosphere_weight * self.atmosphere,
            )
        )
        matte *= 0.25 + 0.75 * self.confidence
        return np.clip(
            cv2.GaussianBlur(matte.astype(np.float32), (0, 0), sigmaX=1.2), 0.0, 1.0
        )


class SceneLayout(BaseModel):
    """Boolean semantic layout used to verify candidates against the source."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    protected: np.ndarray
    water: np.ndarray
    sky: np.ndarray


class SceneDecomposer(Protocol):
    def decompose(self, image: Image.Image) -> SceneDecomposition: ...

    def layout(self, image: Image.Image) -> SceneLayout: ...


class HeuristicSceneDecomposer:
    """Weight-free fallback used for tests and offline pipeline demonstrations."""

    def decompose(self, image: Image.Image) -> SceneDecomposition:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        rgb_float = rgb.astype(np.float32) / 255.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 70, 160)

        red, green, blue = np.moveaxis(rgb_float, -1, 0)
        chroma = rgb_float.max(axis=2) - rgb_float.min(axis=2)
        brightness = rgb_float.mean(axis=2)
        vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        upper_frame = np.broadcast_to(vertical < 0.68, (height, width))
        blue_sky = (blue > red * 1.05) & (blue > green * 0.92) & (brightness > 0.30)
        cloudy_sky = (chroma < 0.16) & (brightness > 0.48)
        candidate = upper_frame & (blue_sky | cloudy_sky) & (edges == 0)
        sky = _top_connected(candidate)

        if sky.mean() < 0.02:
            fallback = np.clip(1.0 - vertical * 4.0, 0.0, 1.0)
            sky = np.broadcast_to(fallback, (height, width)).copy()
            base_confidence = 0.25
        else:
            sky = cv2.GaussianBlur(sky.astype(np.float32), (0, 0), sigmaX=1.5)
            base_confidence = 0.55

        atmosphere = np.broadcast_to((1.0 - vertical) ** 1.5, (height, width)).copy()
        surface = np.broadcast_to(
            np.clip((vertical - 0.42) / 0.58, 0.0, 1.0), (height, width)
        ).copy()
        structure = _build_structure_guard(edges, sky)
        confidence = np.full((height, width), base_confidence, dtype=np.float32)
        confidence = np.maximum(confidence, 0.70 * sky)

        return SceneDecomposition(
            sky=np.clip(sky, 0.0, 1.0).astype(np.float32),
            atmosphere=atmosphere.astype(np.float32),
            weather_surface=surface.astype(np.float32),
            structure_guard=structure,
            confidence=confidence,
        )

    def layout(self, image: Image.Image) -> SceneLayout:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        height, width = rgb.shape[:2]
        red, green, blue = np.moveaxis(rgb, -1, 0)
        brightness = rgb.mean(axis=2)
        vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        blueish = (blue > red * 1.05) & (blue > green * 0.92) & (brightness > 0.25)
        sky = blueish & np.broadcast_to(vertical < 0.5, (height, width))
        water = blueish & np.broadcast_to(vertical >= 0.5, (height, width))
        return SceneLayout(
            protected=np.zeros((height, width), dtype=bool),
            water=water,
            sky=sky,
        )


class TransformerSceneDecomposer:
    """SegFormer plus Depth Anything V2 decomposition with lazy weight loading."""

    def __init__(
        self,
        segmentation_model: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str | None = None,
    ) -> None:

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.segmentation_processor = AutoImageProcessor.from_pretrained(
            segmentation_model
        )
        self.segmentation_model = AutoModelForSemanticSegmentation.from_pretrained(
            segmentation_model
        ).to(self.device)
        self.depth_processor = AutoImageProcessor.from_pretrained(depth_model)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model).to(
            self.device
        )

    def decompose(self, image: Image.Image) -> SceneDecomposition:
        torch = self._torch
        rgb_image = image.convert("RGB")
        height, width = rgb_image.height, rgb_image.width

        labels_np, confidence_np = self._segment(rgb_image)
        with torch.inference_mode():
            depth_inputs = self.depth_processor(
                images=rgb_image, return_tensors="pt"
            ).to(self.device)
            depth = self.depth_model(**depth_inputs).predicted_depth.unsqueeze(1)
            depth = torch.nn.functional.interpolate(
                depth, size=(height, width), mode="bicubic", align_corners=False
            )[0, 0]

        depth_np = depth.cpu().numpy().astype(np.float32)
        id2label = self.segmentation_model.config.id2label

        sky = np.isin(labels_np, _label_ids(id2label, {"sky"})).astype(np.float32)
        weather_surface = np.isin(
            labels_np, _label_ids(id2label, WEATHER_SURFACE_NAMES)
        ).astype(np.float32)
        water = np.isin(labels_np, _label_ids(id2label, WATER_NAMES)).astype(np.float32)
        protected = np.isin(labels_np, _label_ids(id2label, PROTECTED_NAMES)).astype(
            np.uint8
        )

        far_likelihood = _far_likelihood(depth_np, sky)
        atmosphere = cv2.GaussianBlur(far_likelihood, (0, 0), sigmaX=2.0)
        rgb = np.asarray(rgb_image, dtype=np.uint8)
        edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 70, 160)
        transient_texture = np.maximum(sky, water)
        structure = _build_structure_guard(edges, transient_texture, protected)

        return SceneDecomposition(
            sky=cv2.GaussianBlur(sky, (0, 0), sigmaX=1.0),
            atmosphere=np.clip(atmosphere, 0.0, 1.0).astype(np.float32),
            weather_surface=cv2.GaussianBlur(weather_surface, (0, 0), sigmaX=1.0),
            structure_guard=structure,
            confidence=confidence_np,
        )

    def layout(self, image: Image.Image) -> SceneLayout:
        labels_np, _ = self._segment(image.convert("RGB"))
        id2label = self.segmentation_model.config.id2label
        return SceneLayout(
            protected=np.isin(labels_np, _label_ids(id2label, PROTECTED_NAMES)),
            water=np.isin(labels_np, _label_ids(id2label, WATER_NAMES)),
            sky=np.isin(labels_np, _label_ids(id2label, {"sky"})),
        )

    def _segment(self, rgb_image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        with torch.inference_mode():
            seg_inputs = self.segmentation_processor(
                images=rgb_image, return_tensors="pt"
            ).to(self.device)
            seg_logits = self.segmentation_model(**seg_inputs).logits
            seg_logits = torch.nn.functional.interpolate(
                seg_logits,
                size=(rgb_image.height, rgb_image.width),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = seg_logits.softmax(dim=1)[0]
            confidence, labels = probabilities.max(dim=0)
        return labels.cpu().numpy(), confidence.cpu().numpy().astype(np.float32)


def _top_connected(mask: np.ndarray) -> np.ndarray:
    count, components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros(mask.shape, dtype=np.float32)
    touching = np.unique(components[: max(2, mask.shape[0] // 100), :])
    touching = touching[touching != 0]
    return np.isin(components, touching).astype(np.float32)


def _label_ids(id2label: dict[int, str], names: set[str]) -> list[int]:
    normalized = {name.casefold() for name in names}
    return [
        int(label_id)
        for label_id, label in id2label.items()
        if label.casefold() in normalized
    ]


def _build_structure_guard(
    edges: np.ndarray,
    transient_texture: np.ndarray,
    protected: np.ndarray | None = None,
) -> np.ndarray:
    transient_interior = cv2.erode(
        (transient_texture > 0.5).astype(np.uint8),
        np.ones((5, 5), np.uint8),
        iterations=1,
    )
    static_edges = edges.copy()
    static_edges[transient_interior > 0] = 0
    guard = (
        cv2.dilate(static_edges, np.ones((3, 3), np.uint8), iterations=1).astype(
            np.float32
        )
        / 255.0
    )
    if protected is not None:
        guard = np.maximum(guard, protected)
    return guard.astype(np.float32)


def _far_likelihood(depth: np.ndarray, sky: np.ndarray) -> np.ndarray:
    low, high = np.percentile(depth, (2.0, 98.0))
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    non_sky = sky < 0.5
    if sky.any() and non_sky.any():
        sky_is_high = np.median(normalized[sky > 0.5]) >= np.median(normalized[non_sky])
        return normalized if sky_is_high else 1.0 - normalized
    return 1.0 - normalized
