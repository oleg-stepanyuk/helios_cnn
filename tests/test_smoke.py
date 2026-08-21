"""Smoke test: synthetic disk + bright blobs end-to-end.

Generates a 512x512 image of a uniform solar disk with a handful of bright
Gaussian "structures" superimposed, then runs the full pipeline and asserts:

* outputs exist on disk
* at least one structure is recovered
* the combined mask is non-empty

Run with ``pytest`` or directly as ``python tests/test_smoke.py``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from wavetrack import Config, run


def _make_synthetic_image(ny: int = 512, nx: int = 512, rng_seed: int = 42) -> np.ndarray:
    """Synthetic disk with bright blobs + Gaussian noise."""
    rng = np.random.default_rng(rng_seed)
    image = np.zeros((ny, nx), dtype=np.float64)

    # Background disk.
    cy, cx = ny / 2, nx / 2
    yy, xx = np.mgrid[:ny, :nx]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    disk = (rr < 220).astype(np.float64) * 100.0
    image += disk

    # Bright blobs (the "structures" we want to recover).
    blob_centers = [(150, 150), (380, 200), (260, 380), (320, 330)]
    for by, bx in blob_centers:
        sigma = 18.0
        g = np.exp(-((yy - by) ** 2 + (xx - bx) ** 2) / (2 * sigma ** 2))
        image += 220.0 * g

    # Noise.
    image += rng.normal(0.0, 3.0, size=image.shape)
    return image


def _make_config(workdir: Path) -> Config:
    cfg = Config()
    cfg.io.output_dir = str(workdir / "out")
    cfg.io.save_intermediate = True
    cfg.io.save_fits = False
    cfg.preprocess.invert = False
    cfg.preprocess.resolution_bits = 8
    cfg.wavelet.n_scales = 5
    cfg.wavelet.scales_weights = [0.0, 1.0, 1.0, 1.0, 0.0]
    cfg.wavelet.reference_scale = 1
    cfg.threshold.split_disk_limb = False
    cfg.threshold.sigma_disk = 3.0
    cfg.threshold.sigma_grad = 3.0
    cfg.segmentation.grad_method = "sobel-feldman"
    cfg.segmentation.min_struct_size = 50
    cfg.segmentation.window_filt_boxsize = 0  # disable window filter for the test
    cfg.validate()
    return cfg


def run_smoke_test() -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="wavetrack_smoke_"))
    try:
        image = _make_synthetic_image()
        npy_path = workdir / "synthetic.npy"
        np.save(npy_path, image)

        cfg = _make_config(workdir)
        result = run(str(npy_path), cfg)

        # 1) The pipeline returned a sensible result object.
        assert result.image.shape == image.shape
        assert result.decomposition.coefficients[0].shape == image.shape
        assert len(result.decomposition.coefficients) == cfg.wavelet.n_scales

        # 2) Mask is non-empty.
        mask = result.segmentation.combined_mask
        n_on = int(mask.sum())
        assert n_on > 0, "combined mask is empty"

        # 3) At least 1 structure recovered.
        assert result.n_structures >= 1, "no structures detected"

        # 4) Output files exist.
        out_dir = result.output_dir
        for rel in [
            "masks/synthetic_mask.png",
            "masks/synthetic_mask_raw.png",
            "visualize/synthetic_reconstructed.png",
            "visualize/synthetic_gradient.png",
            "visualize/synthetic_preprocessed.png",
            "scales/synthetic_coeff_0.png",
            "scales/synthetic_reference_mask.png",
        ]:
            assert (out_dir / rel).exists(), f"missing output: {rel}"

        # 5) CSV summary written.
        csv = out_dir / "synthetic_structures.csv"
        assert csv.exists()
        text = csv.read_text(encoding="utf-8").strip().splitlines()
        assert text[0].startswith("label,size,")
        assert len(text) >= 2  # header + at least one row

        print(
            f"SMOKE OK: {result.n_structures} structures, "
            f"{n_on} mask pixels, outputs at {out_dir}"
        )
        return out_dir
    finally:
        # Keep outputs only when running interactively (env var WAVETRACK_KEEP=1).
        import os
        if not os.environ.get("WAVETRACK_KEEP"):
            shutil.rmtree(workdir, ignore_errors=True)


def test_smoke() -> None:  # pytest entry point
    run_smoke_test()


if __name__ == "__main__":
    run_smoke_test()
