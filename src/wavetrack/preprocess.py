"""Image preprocessing utilities used before wavelet decomposition.

Implements (in order):

1. Absolute clipping to ``[low_threshold, high_threshold]``.
2. Optional logarithmic dynamic-range compression.
3. Optional image inversion (numeric or PIL-style).
4. Optional posterization / N-bit normalization.
5. Optional circular (solar disk) and rectangular (ROI) masks.

The function :func:`preprocess` returns the preprocessed image together
with the binary disk and ROI masks (when requested), so downstream code
can perform on-disk / off-limb statistics splitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from .config import PreprocessConfig

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """Bundle of arrays produced by :func:`preprocess`."""

    image: np.ndarray
    """Preprocessed image used as input to the wavelet decomposition."""

    original: np.ndarray
    """The original clipped image (no inversion / no normalization)."""

    disk_mask: Optional[np.ndarray]
    """Binary mask (1 inside disk, 0 outside). ``None`` if not requested."""

    roi_mask: Optional[np.ndarray]
    """Binary mask for the rectangular ROI. ``None`` if not requested."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess(image: np.ndarray, cfg: PreprocessConfig) -> PreprocessResult:
    """Run the preprocessing chain on ``image`` according to ``cfg``."""
    arr = np.asarray(image, dtype=np.float64)

    # 1. Clip absolute values.
    arr = _clip(arr, cfg.low_threshold, cfg.high_threshold)
    original = arr.copy()

    # 2. Optional log compression (applied before inversion to operate on
    # physical intensities).
    if cfg.log_compress:
        arr = _log_compress(arr, base=cfg.log_base)

    # 3. Optional inversion.
    if cfg.invert:
        arr = _invert(arr, method=cfg.invert_method)

    # 4. Optional N-bit posterization / normalization.
    if cfg.resolution_bits > 0:
        arr = posterize(arr, bits=cfg.resolution_bits)

    # 5. Disk / ROI masks.
    disk_mask = None
    if cfg.apply_circle_mask:
        ny, nx = arr.shape
        cx = cfg.disk_center_x if cfg.disk_center_x is not None else nx // 2
        cy = cfg.disk_center_y if cfg.disk_center_y is not None else ny // 2
        radius = int(round(float(cfg.disk_radius) * cfg.rsun_coeff))
        disk_mask = circle_mask(arr.shape, center=(cx, cy), radius=radius)

    roi_mask = None
    if cfg.apply_square_mask:
        roi_mask = square_mask(
            arr.shape,
            x1=cfg.square_x1,
            x2=cfg.square_x2,
            y1=cfg.square_y1,
            y2=cfg.square_y2,
        )
        arr = arr * roi_mask

    logger.info(
        "Preprocess: shape=%s min=%.3g max=%.3g mean=%.3g",
        arr.shape,
        np.min(arr),
        np.max(arr),
        float(np.mean(arr)),
    )
    return PreprocessResult(
        image=arr,
        original=original,
        disk_mask=disk_mask,
        roi_mask=roi_mask,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip(image: np.ndarray, low: float, high: float) -> np.ndarray:
    """Zero out pixels outside ``[low, high]``."""
    mask = (image > low) & (image < high)
    return image * mask


def _log_compress(image: np.ndarray, base: float = 20.0) -> np.ndarray:
    """Apply log compression to positive pixels.

    The formula is ``log(1 + base * (x - x_min) / (x_max - x_min))`` so
    constant images become zero rather than raising.
    """
    x_min = float(np.min(image))
    x_max = float(np.max(image))
    if x_max <= x_min:
        return np.zeros_like(image)
    norm = (image - x_min) / (x_max - x_min)
    scale = max(float(base), 1.0)
    return np.log1p(scale * norm)


def _invert(image: np.ndarray, method: str = "numeric") -> np.ndarray:
    """Invert an image either numerically or via PIL."""
    if method == "numeric":
        x_max = float(np.max(image))
        return x_max - image
    if method == "pil":
        # PIL inversion only works on 8-bit; rescale, invert, rescale back.
        x_min = float(np.min(image))
        x_max = float(np.max(image))
        if x_max <= x_min:
            return np.zeros_like(image)
        norm = (image - x_min) / (x_max - x_min)
        as_uint8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        inverted = np.asarray(ImageOps.invert(Image.fromarray(as_uint8)))
        return inverted.astype(np.float64) / 255.0 * (x_max - x_min) + x_min
    raise ValueError(f"Unknown invert method: {method!r}")


def posterize(image: np.ndarray, bits: int) -> np.ndarray:
    """Quantize ``image`` to ``2**bits`` levels mapped onto ``int32`` range.

    Mirrors the original Wavetrack behaviour: data is first compressed to
    ``2**bits`` discrete levels, then expanded to the full int32 dynamic
    range to keep enough resolution for subsequent statistical thresholding.
    """
    x_min = float(np.min(image))
    x_max = float(np.max(image))
    if x_max <= x_min:
        return np.zeros_like(image, dtype=np.float64)

    levels = float(2 ** bits)
    int32_max = float(np.iinfo(np.int32).max)

    norm = (image - x_min) / (x_max - x_min)
    quantized = np.round(norm * (levels - 1))
    out = quantized * (int32_max / (levels - 1))
    return out.astype(np.float64)


def circle_mask(
    shape: Tuple[int, int],
    center: Tuple[int, int],
    radius: int,
) -> np.ndarray:
    """Binary mask: 1 inside a circle of given center and radius, 0 outside."""
    ny, nx = shape
    cx, cy = center
    y, x = np.ogrid[:ny, :nx]
    return ((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2).astype(np.uint8)


def square_mask(
    shape: Tuple[int, int],
    x1: int,
    x2: int,
    y1: int,
    y2: int,
) -> np.ndarray:
    """Binary mask: 1 inside the rectangle ``[x1, x2] x [y1, y2]``, 0 outside.

    Coordinates are clipped to image bounds.
    """
    ny, nx = shape
    mask = np.zeros(shape, dtype=np.uint8)
    x1, x2 = max(0, min(x1, x2)), min(nx, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(ny, max(y1, y2))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask
