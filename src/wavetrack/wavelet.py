"""À-trous wavelet decomposition with per-scale thresholding.

This is the core of the Wavetrack algorithm as described in
Stepanyuk et al. 2022 (J. Space Weather Space Clim. 12, 20). The
processing chain implemented here is:

1. Compute J wavelet coefficients ``c_j`` by iteratively convolving the
   input image with dilated versions of the à-trous mother kernel ``k_0``.
   At iteration ``j``, the kernel is dilated by inserting ``2**(j-1) - 1``
   zeros between each row and column of ``k_0`` ("a trous" = "with holes").
2. Apply a relative (sigma-based) threshold to every coefficient
   ``c_j`` for ``j >= 1``. When a disk mask is provided, on-disk and
   off-limb pixels are thresholded independently and merged, to handle
   the very different statistical distributions of those two regions.
3. Compute the wavelet scales ``omega_j = c_{j-1} - c_j``. Each scale
   isolates a specific detail level of the image.
4. Reconstruct a filtered image by summing the (per-scale weighted)
   scales together with the coarsest coefficient ``c_{J-1}``.
5. Apply a binary "reference mask" from a chosen coarse coefficient to
   the reconstructed image; this restricts detected features to the
   spatial support of that coarse scale.

The processing-scale-by-scale-and-reassemble logic is the key feature of
the method and is preserved verbatim from the original implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy import ndimage
from scipy.stats import norm

from .config import ThresholdConfig, WaveletConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mother kernel (à-trous, as published in Stepanyuk et al. 2022, Eq. A12)
# ---------------------------------------------------------------------------

A_TROUS_KERNEL = np.array(
    [
        [1 / 256, 1 / 64, 3 / 128, 1 / 64, 1 / 256],
        [1 / 64, 1 / 16, 3 / 32, 1 / 16, 1 / 64],
        [3 / 128, 3 / 32, 9 / 64, 3 / 32, 3 / 128],
        [1 / 64, 1 / 16, 3 / 32, 1 / 16, 1 / 64],
        [1 / 256, 1 / 64, 3 / 128, 1 / 64, 1 / 256],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Decomposition result
# ---------------------------------------------------------------------------


@dataclass
class WaveletDecomposition:
    """Holds the outputs of :func:`decompose`."""

    coefficients: List[np.ndarray] = field(default_factory=list)
    """Wavelet coefficients ``c_0..c_{J-1}`` after sigma-thresholding
    (``c_0`` is the original image, untouched)."""

    scales: List[np.ndarray] = field(default_factory=list)
    """Wavelet scales ``omega_j = c_{j-1} - c_j`` for ``j>=1``.
    ``scales[0]`` is a copy of the input image to keep indexing symmetric
    with ``coefficients``."""

    reconstructed: Optional[np.ndarray] = None
    """Image reconstructed from weighted scales plus the coarsest
    coefficient."""

    reconstructed_masked: Optional[np.ndarray] = None
    """Reconstructed image after applying the reference-scale binary mask."""

    reference_mask: Optional[np.ndarray] = None
    """Binary mask derived from the chosen reference coefficient."""

    stds: List[float] = field(default_factory=list)
    """Fitted Gaussian standard deviation of every coefficient (before
    thresholding). Useful for logging / parameter tuning."""


# ---------------------------------------------------------------------------
# Kernel dilation
# ---------------------------------------------------------------------------


def dilate_kernel(kernel: np.ndarray, j: int) -> np.ndarray:
    """Return the j-th à-trous dilation of ``kernel``.

    For ``j == 1`` this returns ``kernel`` unchanged. For ``j > 1``,
    ``2**(j-1) - 1`` zeros are inserted between consecutive rows/columns of
    the original kernel.
    """
    if j < 1:
        raise ValueError("j must be >= 1")
    if j == 1:
        return np.ascontiguousarray(kernel, dtype=np.float64)

    gap = (2 ** (j - 1)) - 1  # number of zeros between original entries
    step = gap + 1  # stride from one original entry to the next
    rows, cols = kernel.shape
    out_shape = ((rows - 1) * step + 1, (cols - 1) * step + 1)
    dilated = np.zeros(out_shape, dtype=np.float64)
    dilated[::step, ::step] = kernel
    return dilated


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def decompose(
    image: np.ndarray,
    wavelet_cfg: WaveletConfig,
    threshold_cfg: ThresholdConfig,
    disk_mask: Optional[np.ndarray] = None,
) -> WaveletDecomposition:
    """Run the full à-trous decomposition + reconstruction pipeline.

    Parameters
    ----------
    image
        Preprocessed 2-D image array.
    wavelet_cfg
        Wavelet decomposition parameters (number of scales, weights,
        reference scale index).
    threshold_cfg
        Per-coefficient sigma thresholding parameters.
    disk_mask
        Optional binary mask of the solar disk. When provided AND
        ``threshold_cfg.split_disk_limb`` is true, statistics are computed
        separately on-disk and off-limb.

    Returns
    -------
    WaveletDecomposition
    """
    image = np.asarray(image, dtype=np.float64)
    j_max = wavelet_cfg.n_scales

    if threshold_cfg.split_disk_limb and disk_mask is None:
        raise ValueError(
            "threshold.split_disk_limb=true requires a disk mask "
            "(set preprocess.apply_circle_mask=true)"
        )

    # 1) Compute wavelet coefficients c_0..c_{J-1}.
    coefficients, stds = _compute_coefficients(
        image, j_max=j_max, threshold_cfg=threshold_cfg, disk_mask=disk_mask
    )

    # 2) Compute scales omega_j = c_{j-1} - c_j  for j = 1..J-1.
    scales: List[np.ndarray] = [image.copy()]
    for j in range(1, j_max):
        scales.append(coefficients[j - 1].astype(np.float64) - coefficients[j].astype(np.float64))

    # 3) Reconstruct: start from the coarsest coefficient and add weighted scales.
    weights = list(wavelet_cfg.scales_weights)
    reconstructed = coefficients[j_max - 1].astype(np.float64).copy()
    for j in range(1, j_max):
        reconstructed = reconstructed + weights[j] * scales[j]

    # 4) Build the reference mask from the chosen coefficient and apply it.
    ref_idx = wavelet_cfg.reference_scale
    reference_arr = coefficients[ref_idx]
    reference_mask = (reference_arr > 0).astype(np.uint8)
    reconstructed_masked = reconstructed * reference_mask

    logger.info(
        "Decomposition complete: J=%d, reference=c_%d (active pixels=%d)",
        j_max,
        ref_idx,
        int(reference_mask.sum()),
    )

    return WaveletDecomposition(
        coefficients=coefficients,
        scales=scales,
        reconstructed=reconstructed,
        reconstructed_masked=reconstructed_masked,
        reference_mask=reference_mask,
        stds=stds,
    )


# ---------------------------------------------------------------------------
# Coefficient computation with per-scale thresholding
# ---------------------------------------------------------------------------


def _compute_coefficients(
    image: np.ndarray,
    j_max: int,
    threshold_cfg: ThresholdConfig,
    disk_mask: Optional[np.ndarray],
) -> tuple[List[np.ndarray], List[float]]:
    """Iteratively convolve image with dilated kernels and threshold.

    Returns the list of (thresholded) coefficients and the fitted std for
    each coefficient.
    """
    coefficients: List[np.ndarray] = [image.copy()]
    _, std_0 = norm.fit(image.ravel())
    stds: List[float] = [float(std_0)]

    for j in range(1, j_max):
        kernel_j = dilate_kernel(A_TROUS_KERNEL, j)
        c_j = ndimage.convolve(
            coefficients[j - 1].astype(np.float64),
            kernel_j,
            mode="reflect",
        )

        if threshold_cfg.split_disk_limb and disk_mask is not None:
            c_j_thr, std_j = _threshold_split(
                c_j,
                disk_mask=disk_mask,
                sigma_disk=threshold_cfg.sigma_disk,
                sigma_limb=threshold_cfg.sigma_limb,
            )
        else:
            _, std_j = norm.fit(c_j.ravel())
            std_j = float(std_j)
            thr = threshold_cfg.sigma_disk * std_j
            c_j_thr = c_j * (c_j > thr)

        logger.info(
            "  c_%d: kernel %sx%s, std=%.4g",
            j,
            kernel_j.shape[0],
            kernel_j.shape[1],
            std_j,
        )
        coefficients.append(c_j_thr)
        stds.append(float(std_j))

    return coefficients, stds


def _threshold_split(
    coeff: np.ndarray,
    disk_mask: np.ndarray,
    sigma_disk: float,
    sigma_limb: float,
) -> tuple[np.ndarray, float]:
    """Threshold ``coeff`` independently on-disk and off-limb, then merge.

    Returns the merged thresholded array and the on-disk fitted std (used
    as a representative summary value for logging).
    """
    inside = coeff * (disk_mask > 0)
    outside = coeff * (disk_mask == 0)

    _, std_in = norm.fit(inside.ravel())
    _, std_out = norm.fit(outside.ravel())

    inside_thr = inside * (inside > sigma_disk * std_in)
    outside_thr = outside * (outside > sigma_limb * std_out)
    merged = inside_thr + outside_thr
    return merged, float(std_in)


# ---------------------------------------------------------------------------
# Convenience: reconstruct from already-computed scales
# ---------------------------------------------------------------------------


def reconstruct_from_scales(
    coarse_coefficient: np.ndarray,
    scales: Sequence[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    """Recompose an image as ``c_{J-1} + sum_j weights[j] * scales[j]``."""
    out = coarse_coefficient.astype(np.float64).copy()
    for j, scale in enumerate(scales):
        if j == 0:
            continue  # scales[0] is a copy of the original image
        if j >= len(weights):
            break
        out = out + float(weights[j]) * scale.astype(np.float64)
    return out
