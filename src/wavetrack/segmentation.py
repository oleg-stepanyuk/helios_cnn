"""Object/structure segmentation on top of the wavelet reconstruction.

The segmentation routine follows Pathway A from Stepanyuk et al. 2022:

1. Binarize the (already-filtered) reconstructed image.
2. Label connected components.
3. Drop components smaller than ``min_struct_size`` pixels.
4. Compute the center of mass (intensity-weighted) and the geometric
   center for every surviving structure.
5. Apply a one-pass window filter that suppresses sub-window patches
   below threshold, cleaning up small spurious features.
6. Build a single combined mask from all surviving structures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy import ndimage

from .config import SegmentationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class Structure:
    """A single connected component extracted from the reconstruction."""

    label: int
    mask: np.ndarray
    """Binary mask of this structure (uint8)."""
    raw: np.ndarray
    """``raw_image * mask`` — intensity values inside the structure."""
    size: int
    """Number of pixels in the structure."""
    center_of_mass: Tuple[float, float]
    """Intensity-weighted center ``(x, y)`` in pixel coordinates."""
    geometric_center: Tuple[float, float]
    """Center of the binary mask ``(x, y)``."""


@dataclass
class SegmentationResult:
    """Aggregated output of :func:`segment`."""

    structures: List[Structure] = field(default_factory=list)
    combined_mask: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    combined_raw: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    n_structures: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment(
    reconstructed: np.ndarray,
    raw_for_masking: np.ndarray,
    cfg: SegmentationConfig,
) -> SegmentationResult:
    """Segment objects from a reconstructed image.

    Parameters
    ----------
    reconstructed
        Image to threshold and binarize (typically the output of
        :func:`wavetrack.wavelet.decompose`'s ``reconstructed_masked``).
    raw_for_masking
        Image whose intensities are reported inside each structure mask
        (typically the preprocessed image itself or the original data).
    cfg
        Segmentation configuration.
    """
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    raw = np.asarray(raw_for_masking, dtype=np.float64)

    if reconstructed.shape != raw.shape:
        raise ValueError(
            f"reconstructed shape {reconstructed.shape} must match raw "
            f"shape {raw.shape}"
        )

    # 1) Binarize. Anything non-zero counts: thresholding was already
    # applied during wavelet decomposition.
    binary = (reconstructed != 0).astype(np.uint8)

    # 2) Label connected components (8-connectivity).
    structure_elem = np.ones((3, 3), dtype=np.uint8)
    labels, n_labels = ndimage.label(binary, structure=structure_elem)
    logger.info("Connected-component labeling: %d candidates", n_labels)

    structures: List[Structure] = []
    combined_mask = np.zeros_like(binary)

    # 3) Filter by size, build Structure objects.
    sizes = ndimage.sum_labels(np.ones_like(binary), labels, range(1, n_labels + 1))
    for idx, size in enumerate(sizes, start=1):
        size_int = int(size)
        if size_int < cfg.min_struct_size:
            continue

        mask = (labels == idx).astype(np.uint8)
        raw_in_mask = raw * mask
        com = _center_of_mass(raw_in_mask)
        gcom = _center_of_mass(mask.astype(np.float64))

        structures.append(
            Structure(
                label=idx,
                mask=mask,
                raw=raw_in_mask,
                size=size_int,
                center_of_mass=com,
                geometric_center=gcom,
            )
        )
        combined_mask = combined_mask | mask

    logger.info(
        "Kept %d structures (min size = %d px)",
        len(structures),
        cfg.min_struct_size,
    )

    # 4) Final window filter on the combined mask.
    if cfg.window_filt_boxsize > 1:
        combined_mask = window_filter(
            combined_mask,
            box_size=cfg.window_filt_boxsize,
            increment=cfg.window_filt_increment,
            threshold=cfg.window_filt_threshold,
        )

    combined_raw = combined_mask * raw

    return SegmentationResult(
        structures=structures,
        combined_mask=combined_mask,
        combined_raw=combined_raw,
        n_structures=len(structures),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _center_of_mass(weights: np.ndarray) -> Tuple[float, float]:
    """Return the intensity-weighted center as ``(x, y)``.

    For a binary mask, this is equivalent to the geometric center.
    Returns ``(0.0, 0.0)`` if the array sums to zero.
    """
    arr = np.asarray(weights, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0:
        return (0.0, 0.0)
    ny, nx = arr.shape
    y_grid, x_grid = np.mgrid[:ny, :nx]
    x_cent = float((x_grid * arr).sum() / total)
    y_cent = float((y_grid * arr).sum() / total)
    return (x_cent, y_cent)


try:
    from numba import njit  # type: ignore

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False


if _HAVE_NUMBA:

    @njit(cache=True, fastmath=False)
    def _window_filter_numba(arr: np.ndarray, box_size: int, increment: int, threshold: float) -> np.ndarray:
        ny, nx = arr.shape
        for y in range(1, ny - box_size, increment):
            for x in range(1, nx - box_size, increment):
                y2 = y + box_size
                x2 = x + box_size
                tmax = arr[y, x]
                for k in range(x + 1, x2):
                    if arr[y, k] > tmax:
                        tmax = arr[y, k]
                if tmax > threshold:
                    continue
                bmax = arr[y2, x]
                for k in range(x + 1, x2):
                    if arr[y2, k] > bmax:
                        bmax = arr[y2, k]
                if bmax > threshold:
                    continue
                lmax = arr[y, x]
                for k in range(y + 1, y2):
                    if arr[k, x] > lmax:
                        lmax = arr[k, x]
                if lmax > threshold:
                    continue
                rmax = arr[y, x2]
                for k in range(y + 1, y2):
                    if arr[k, x2] > rmax:
                        rmax = arr[k, x2]
                if rmax > threshold:
                    continue
                for yy in range(y, y2):
                    for xx in range(x, x2):
                        arr[yy, xx] = 0.0
        return arr


def window_filter(
    mask: np.ndarray,
    box_size: int = 10,
    increment: int = 1,
    threshold: float = 0.0,
) -> np.ndarray:
    """Suppress small spurious patches in ``mask``.

    Iterates a sliding window across the image; if all four borders of
    the window are below ``threshold`` the window's interior is zeroed.
    This mirrors the final cleanup pass in the original implementation.

    A Numba-accelerated path is used when available; falls back to the
    pure-NumPy implementation otherwise.
    """
    arr = np.array(mask, copy=True)
    ny, nx = arr.shape

    if box_size < 2 or increment < 1:
        return arr

    if _HAVE_NUMBA:
        arr64 = arr.astype(np.float64, copy=True)
        _window_filter_numba(arr64, box_size, increment, float(threshold))
        return arr64.astype(arr.dtype, copy=False)

    # Pure-Python fallback (slow).
    for y in range(1, ny - box_size, increment):
        for x in range(1, nx - box_size, increment):
            y2 = y + box_size
            x2 = x + box_size
            top = arr[y, x:x2]
            bottom = arr[y2, x:x2]
            left = arr[y:y2, x]
            right = arr[y:y2, x2]
            if (
                top.max() <= threshold
                and bottom.max() <= threshold
                and left.max() <= threshold
                and right.max() <= threshold
            ):
                arr[y:y2, x:x2] = 0
    return arr
