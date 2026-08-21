"""Gradient and curvature operators used as structural criteria.

The original Wavetrack code supports three methods:

* **Sobel-Feldman** (default): discrete approximation of the image
  gradient magnitude using the classic 3x3 Sobel-Feldman kernel pair.
* **Simple gradient**: convolution with a small ``[-1, 0, 1]`` kernel
  along both axes.
* **Hagenaer curvature**: a curvature-based "saddle point" detector
  defined in Hagenaer et al. 1999 (Astrophys. J. 511, 932).

A gradient field highlights object contours, allows saddle-point
detection and serves as an additional criterion during segmentation. It
is also useful as an extra channel for deep-learning training sets.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

SOBEL_GX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
SOBEL_GY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)

SIMPLE_GX = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=np.float64)
SIMPLE_GY = SIMPLE_GX.T


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_gradient(
    image: np.ndarray, method: str = "sobel-feldman"
) -> np.ndarray:
    """Return the gradient magnitude (or curvature) of ``image``.

    Parameters
    ----------
    image
        2-D input array.
    method
        ``'sobel-feldman'``, ``'simple'``, ``'hagenaer'`` or ``'none'``.
        ``'none'`` returns a zero array, useful to disable the gradient
        criterion entirely while keeping a uniform pipeline.

    Returns
    -------
    grad : ndarray
        Same shape as ``image``, ``float64``.
    """
    method = method.lower()
    arr = np.asarray(image, dtype=np.float64)

    if method == "none":
        return np.zeros_like(arr)
    if method == "sobel-feldman":
        return _convolve_magnitude(arr, SOBEL_GX, SOBEL_GY)
    if method == "simple":
        return _convolve_magnitude(arr, SIMPLE_GX, SIMPLE_GY)
    if method == "hagenaer":
        return hagenaer_curvature(arr)
    raise ValueError(f"Unknown gradient method: {method!r}")


def gradient_angle(image: np.ndarray, method: str = "sobel-feldman") -> np.ndarray:
    """Return the per-pixel gradient direction (radians) for ``image``."""
    method = method.lower()
    if method == "sobel-feldman":
        kx, ky = SOBEL_GX, SOBEL_GY
    elif method == "simple":
        kx, ky = SIMPLE_GX, SIMPLE_GY
    else:
        raise ValueError(f"Gradient angle is undefined for method {method!r}")

    arr = np.asarray(image, dtype=np.float64)
    gx = ndimage.convolve(arr, kx, mode="reflect")
    gy = ndimage.convolve(arr, ky, mode="reflect")
    return np.arctan2(gy, gx)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _convolve_magnitude(
    image: np.ndarray, kx: np.ndarray, ky: np.ndarray
) -> np.ndarray:
    gx = ndimage.convolve(image, kx, mode="reflect")
    gy = ndimage.convolve(image, ky, mode="reflect")
    return np.sqrt(gx * gx + gy * gy)


def hagenaer_curvature(image: np.ndarray) -> np.ndarray:
    """Curvature-based saddle-point detector (Hagenaer et al., 1999).

    For each pixel, count how many of the four oriented second-differences
    ``xi(h1, h2) = 2*I(i,j) - I(i+h1, j+h2) - I(i-h1, j-h2)`` are positive.
    A pixel is flagged as belonging to a curvature edge when more than
    three of the directions agree.
    """
    arr = np.asarray(image, dtype=np.float64)
    ny, nx = arr.shape
    counter = np.zeros_like(arr, dtype=np.int32)

    # Four oriented neighbour offsets: (1,0), (1,1), (0,1), (-1,1).
    offsets = [(1, 0), (1, 1), (0, 1), (-1, 1)]
    for h1, h2 in offsets:
        shifted_pos = np.roll(np.roll(arr, -h1, axis=0), -h2, axis=1)
        shifted_neg = np.roll(np.roll(arr, h1, axis=0), h2, axis=1)
        xi = 2.0 * arr - shifted_pos - shifted_neg
        counter += (xi > 0).astype(np.int32)

    # Zero out borders to avoid wrap-around artifacts from np.roll.
    counter[:1, :] = 0
    counter[-1:, :] = 0
    counter[:, :1] = 0
    counter[:, -1:] = 0
    return (counter > 3).astype(np.float64)
