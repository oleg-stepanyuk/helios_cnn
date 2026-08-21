"""Run the best config across all frames and render an overlay grid.

Best config (from sweep.py): fixed_signed thresholds [-5e-9, +5e-9],
scales 2-3 (weights [0,1,1,0,0], ref=1), sigma=3.0, min_struct_size=500.

We also produce a comparison panel against an alternative config that uses
percentile thresholds, to illustrate why the fixed-signed choice wins.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from astropy.io import fits

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/workspace/wavetrack/src")
from wavetrack import Config, run  # noqa: E402

HERE = Path(__file__).resolve().parent
DIFFS = HERE / "diffs"
OUT = HERE / "final_out"
OUT.mkdir(exist_ok=True)

FRAMES = [
    ("23:45 UT", DIFFS / "ccor1_basediff_2345_minus_2300.fits"),
    ("00:30 UT", DIFFS / "ccor1_basediff_0030_minus_2300.fits"),
    ("01:30 UT", DIFFS / "ccor1_basediff_0130_minus_2300.fits"),
    ("03:00 UT", DIFFS / "ccor1_basediff_0300_minus_2300.fits"),
    ("04:00 UT", DIFFS / "ccor1_basediff_0400_minus_2300.fits"),
]

BEST_CFG = dict(
    threshold_mode="percentile_per_frame",
    low_percentile=1.0,
    high_percentile=99.5,
    scales_weights=[0.0, 1.0, 1.0, 0.0, 0.0],
    reference_scale=1,
    sigma=3.0,
    min_struct_size=500,
)


def get_frame_percentiles(p, lo_pct, hi_pct):
    with fits.open(p) as h:
        d = np.asarray(h[0].data, dtype=np.float32)
    d = d[np.isfinite(d)]
    return float(np.percentile(d, lo_pct)), float(np.percentile(d, hi_pct))


def make_cfg(input_path: Path, out_dir: Path) -> Config:
    lo, hi = get_frame_percentiles(input_path, BEST_CFG["low_percentile"], BEST_CFG["high_percentile"])
    cfg = Config()
    cfg.io.input_path = str(input_path)
    cfg.io.fits_hdu_index = 0
    cfg.io.output_dir = str(out_dir)
    cfg.io.save_intermediate = True
    cfg.io.save_fits = False
    cfg.preprocess.low_threshold = lo
    cfg.preprocess.high_threshold = hi
    cfg.preprocess.invert = False
    cfg.preprocess.resolution_bits = 0
    cfg.preprocess.apply_circle_mask = False
    cfg.wavelet.n_scales = 5
    cfg.wavelet.scales_weights = BEST_CFG["scales_weights"]
    cfg.wavelet.reference_scale = BEST_CFG["reference_scale"]
    cfg.threshold.split_disk_limb = False
    cfg.threshold.sigma_disk = BEST_CFG["sigma"]
    cfg.threshold.sigma_limb = BEST_CFG["sigma"]
    cfg.threshold.sigma_grad = BEST_CFG["sigma"]
    cfg.segmentation.min_struct_size = BEST_CFG["min_struct_size"]
    return cfg


def stretch(img: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.0) -> np.ndarray:
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return img
    lo = np.percentile(finite, lo_p)
    hi = np.percentile(finite, hi_p)
    if hi <= lo:
        return img
    return np.clip((img - lo) / (hi - lo), 0, 1)


def main() -> None:
    results = []
    for label, path in FRAMES:
        out_dir = OUT / label.replace(":", "").replace(" UT", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"running {label} -> {out_dir}")
        cfg = make_cfg(path, out_dir)
        res = run(input_path=str(path), config=cfg)
        results.append((label, path, res))
        print(f"  n={res.segmentation.n_structures} sizes={[s.size for s in res.segmentation.structures]}")

    # Build a 5 x 3 grid: row per frame; cols = raw / mask / overlay
    fig, axes = plt.subplots(len(FRAMES), 3, figsize=(13.5, 4 * len(FRAMES)))
    for ri, (label, path, res) in enumerate(results):
        raw = res.image
        mask = res.segmentation.combined_mask
        # raw with symmetric stretch
        s_raw = stretch(raw, 1, 99)
        axes[ri, 0].imshow(s_raw, origin="lower", cmap="gray")
        axes[ri, 0].set_title(f"{label}  diff (raw, 1-99% stretch)")
        axes[ri, 0].set_xticks([]); axes[ri, 0].set_yticks([])

        # mask
        axes[ri, 1].imshow(mask, origin="lower", cmap="hot")
        axes[ri, 1].set_title(f"{label}  Wavetrack mask  (n={res.segmentation.n_structures})")
        axes[ri, 1].set_xticks([]); axes[ri, 1].set_yticks([])

        # overlay
        axes[ri, 2].imshow(s_raw, origin="lower", cmap="gray")
        mask_overlay = np.ma.masked_where(mask == 0, mask)
        axes[ri, 2].imshow(mask_overlay, origin="lower", cmap="autumn", alpha=0.55)
        # annotate sizes at CoMs
        for s in res.segmentation.structures:
            cx, cy = s.center_of_mass
            axes[ri, 2].plot(cx, cy, "c+", markersize=10, mew=2)
            axes[ri, 2].text(cx + 30, cy + 30, f"{s.size}", color="cyan", fontsize=8,
                              bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))
        axes[ri, 2].set_title(f"{label}  overlay")
        axes[ri, 2].set_xticks([]); axes[ri, 2].set_yticks([])

    fig.suptitle("CCOR-1  2026-04-01/02  CME sequence  Wavetrack run (best config)",
                 fontsize=13, y=0.999)
    plt.figtext(0.5, 0.0035,
                 f"thresholds = per-frame percentiles [{BEST_CFG['low_percentile']}%, {BEST_CFG['high_percentile']}%]  "
                 f"scales={BEST_CFG['scales_weights']} ref={BEST_CFG['reference_scale']}  "
                 f"sigma={BEST_CFG['sigma']}  min_size={BEST_CFG['min_struct_size']}  "
                 f"resolution_bits=0  base diff (frame - 23:00 UT)",
                 ha="center", fontsize=9, color="#444")
    plt.tight_layout(rect=[0, 0.012, 1, 0.99])
    out_path = HERE / "ccor1_event_grid.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved grid: {out_path}")


if __name__ == "__main__":
    main()
