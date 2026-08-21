"""End-to-end Wavetrack pipeline for a single image.

Usage
-----
::

    from wavetrack import Config, run

    cfg = Config.from_toml("examples/config.toml")
    result = run("data/img.fits", cfg)
    print(f"Detected {result.segmentation.n_structures} structures")

The pipeline glues together :mod:`wavetrack.io`,
:mod:`wavetrack.preprocess`, :mod:`wavetrack.wavelet`,
:mod:`wavetrack.gradients` and :mod:`wavetrack.segmentation`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import io as wt_io
from .config import Config
from .gradients import compute_gradient
from .preprocess import PreprocessResult, preprocess
from .segmentation import SegmentationResult, segment
from .wavelet import WaveletDecomposition, decompose

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Bundle returned by :func:`run`."""

    image: np.ndarray
    header: dict
    preprocess: PreprocessResult
    decomposition: WaveletDecomposition
    gradient: np.ndarray
    segmentation: SegmentationResult
    output_dir: Path

    @property
    def n_structures(self) -> int:
        return self.segmentation.n_structures


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    input_path: Optional[str] = None,
    config: Optional[Config] = None,
    config_path: Optional[str] = None,
) -> PipelineResult:
    """Run the full pipeline on a single image.

    Either ``config`` or ``config_path`` must be provided. ``input_path``
    overrides ``config.io.input_path`` when supplied.
    """
    if config is None:
        if config_path is None:
            raise ValueError("Either `config` or `config_path` must be provided")
        config = Config.from_toml(config_path)

    image_path = input_path or config.io.input_path
    if not image_path:
        raise ValueError(
            "No input image specified (set `io.input_path` in the config "
            "or pass `input_path` to `run()`)."
        )

    out_dir = wt_io.ensure_dir(config.io.output_dir)
    logger.info("Output directory: %s", out_dir)

    # 1. Load.
    image, header = wt_io.load_image(
        image_path,
        fits_hdu_index=config.io.fits_hdu_index,
        resize_to=config.io.resize_to,
    )

    # 2. Preprocess.
    pre = preprocess(image, config.preprocess)

    # 3. À-trous decomposition + per-scale thresholding + reconstruction.
    decomp = decompose(
        pre.image,
        wavelet_cfg=config.wavelet,
        threshold_cfg=config.threshold,
        disk_mask=pre.disk_mask,
    )

    # 4. Gradient field on the masked reconstruction (used as a saved
    # diagnostic; segmentation already benefits from the wavelet filtering).
    grad = compute_gradient(
        decomp.reconstructed_masked,
        method=config.segmentation.grad_method,
    )

    # 5. Segment objects.
    seg = segment(
        reconstructed=decomp.reconstructed_masked,
        raw_for_masking=pre.original,
        cfg=config.segmentation,
    )

    # 6. Save outputs.
    _save_outputs(
        out_dir=out_dir,
        image_path=Path(image_path),
        config=config,
        pre=pre,
        decomp=decomp,
        grad=grad,
        seg=seg,
        header=header,
    )

    logger.info(
        "Pipeline complete: %d structures saved to %s",
        seg.n_structures,
        out_dir,
    )
    return PipelineResult(
        image=image,
        header=header,
        preprocess=pre,
        decomposition=decomp,
        gradient=grad,
        segmentation=seg,
        output_dir=out_dir,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _save_outputs(
    out_dir: Path,
    image_path: Path,
    config: Config,
    pre: PreprocessResult,
    decomp: WaveletDecomposition,
    grad: np.ndarray,
    seg: SegmentationResult,
    header: dict,
) -> None:
    stem = image_path.stem

    masks_dir = wt_io.ensure_dir(out_dir / "masks")
    viz_dir = wt_io.ensure_dir(out_dir / "visualize")

    # Always save the headline products: combined mask + masked raw +
    # reconstructed image + gradient field.
    wt_io.save_png(seg.combined_mask, masks_dir / f"{stem}_mask.png")
    wt_io.save_png(seg.combined_raw, masks_dir / f"{stem}_mask_raw.png")
    wt_io.save_png(decomp.reconstructed_masked, viz_dir / f"{stem}_reconstructed.png")
    wt_io.save_png(grad, viz_dir / f"{stem}_gradient.png")
    wt_io.save_png(pre.image, viz_dir / f"{stem}_preprocessed.png")

    # Optional FITS outputs (the raw data products useful for ML pipelines).
    if config.io.save_fits:
        wt_io.save_fits(seg.combined_mask, masks_dir / f"{stem}_mask.fits", header)
        wt_io.save_fits(seg.combined_raw, masks_dir / f"{stem}_mask_raw.fits", header)

    # Per-structure separate masks.
    if seg.structures:
        per_struct_dir = wt_io.ensure_dir(masks_dir / "structures")
        for struct in seg.structures:
            wt_io.save_png(
                struct.mask,
                per_struct_dir / f"{stem}_struct_{struct.label:03d}_mask.png",
            )
            wt_io.save_png(
                struct.raw,
                per_struct_dir / f"{stem}_struct_{struct.label:03d}_raw.png",
            )

    # Intermediate wavelet coefficients and scales.
    if config.io.save_intermediate:
        scales_dir = wt_io.ensure_dir(out_dir / "scales")
        for j, c_j in enumerate(decomp.coefficients):
            wt_io.save_png(c_j, scales_dir / f"{stem}_coeff_{j}.png")
        for j, omega_j in enumerate(decomp.scales):
            if j == 0:
                continue
            wt_io.save_png(omega_j, scales_dir / f"{stem}_scale_{j}.png")
        wt_io.save_png(
            decomp.reference_mask, scales_dir / f"{stem}_reference_mask.png"
        )

    # Write a small structures.csv summary so results are easy to load
    # downstream (e.g. into pandas or a Jupyter notebook).
    summary_path = out_dir / f"{stem}_structures.csv"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("label,size,x_com,y_com,x_geom,y_geom\n")
        for s in seg.structures:
            fh.write(
                f"{s.label},{s.size},"
                f"{s.center_of_mass[0]:.3f},{s.center_of_mass[1]:.3f},"
                f"{s.geometric_center[0]:.3f},{s.geometric_center[1]:.3f}\n"
            )
