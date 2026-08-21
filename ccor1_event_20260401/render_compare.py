"""Compare 3 thresholding strategies side-by-side on the peak frame (04:00 UT).

This illustrates why the sweep favoured the symmetric fixed threshold for
the early CME, and what changes at the peak when signal saturates.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/workspace/wavetrack/src")
from wavetrack import Config, run  # noqa: E402

HERE = Path(__file__).resolve().parent
DIFFS = HERE / "diffs"


def get_stats(p):
    with fits.open(p) as h:
        d = np.asarray(h[0].data, dtype=np.float32)
    d = d[np.isfinite(d)]
    return float(np.percentile(d, 1)), float(np.percentile(d, 99.5))


def make_cfg(input_path, out_dir, lo, hi, label):
    cfg = Config()
    cfg.io.input_path = str(input_path)
    cfg.io.fits_hdu_index = 0
    cfg.io.output_dir = str(out_dir)
    cfg.io.save_intermediate = False
    cfg.io.save_fits = False
    cfg.preprocess.low_threshold = lo
    cfg.preprocess.high_threshold = hi
    cfg.preprocess.invert = False
    cfg.preprocess.resolution_bits = 0
    cfg.preprocess.apply_circle_mask = False
    cfg.wavelet.n_scales = 5
    cfg.wavelet.scales_weights = [0.0, 1.0, 1.0, 0.0, 0.0]
    cfg.wavelet.reference_scale = 1
    cfg.threshold.split_disk_limb = False
    cfg.threshold.sigma_disk = 3.0
    cfg.threshold.sigma_limb = 3.0
    cfg.threshold.sigma_grad = 3.0
    cfg.segmentation.min_struct_size = 500
    return cfg


def stretch(img, lo_p=1.0, hi_p=99.0):
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return img
    lo = np.percentile(finite, lo_p)
    hi = np.percentile(finite, hi_p)
    return np.clip((img - lo) / max(hi - lo, 1e-30), 0, 1)


def main():
    targets = [
        ("00:30 UT (early CME)", DIFFS / "ccor1_basediff_0030_minus_2300.fits"),
        ("01:30 UT (mid CME)",  DIFFS / "ccor1_basediff_0130_minus_2300.fits"),
        ("04:00 UT (peak CME)", DIFFS / "ccor1_basediff_0400_minus_2300.fits"),
    ]

    rows = []
    for label, path in targets:
        p1, p995 = get_stats(path)
        # Three strategies
        strategies = [
            ("Fixed   [-5e-9, +5e-9]", -5e-9, 5e-9),
            ("Pos-only  [0, +5e-9]",    0.0,  5e-9),
            (f"Percentile [p1={p1:.1e}, p99.5={p995:.1e}]", p1, p995),
        ]
        row = []
        for slabel, lo, hi in strategies:
            tmp = HERE / "compare_tmp" / f"{label}_{slabel}"
            tmp.mkdir(parents=True, exist_ok=True)
            cfg = make_cfg(path, tmp, lo, hi, slabel)
            res = run(input_path=str(path), config=cfg)
            row.append((slabel, res))
        rows.append((label, path, row))

    fig, axes = plt.subplots(len(rows), len(rows[0][2]), figsize=(15, 4 * len(rows)))
    for ri, (label, path, row) in enumerate(rows):
        with fits.open(path) as hdul:
            raw = np.asarray(hdul[0].data, dtype=np.float32)
        s_raw = stretch(raw, 1, 99)
        for ci, (slabel, res) in enumerate(row):
            axes[ri, ci].imshow(s_raw, origin="lower", cmap="gray")
            mask = res.segmentation.combined_mask
            mo = np.ma.masked_where(mask == 0, mask)
            axes[ri, ci].imshow(mo, origin="lower", cmap="autumn", alpha=0.55)
            for s in res.segmentation.structures:
                cx, cy = s.center_of_mass
                axes[ri, ci].plot(cx, cy, "c+", markersize=8, mew=1.5)
            axes[ri, ci].set_title(
                f"{label}\n{slabel}\nn={res.segmentation.n_structures}, "
                f"px={int(mask.sum())}",
                fontsize=9,
            )
            axes[ri, ci].set_xticks([]); axes[ri, ci].set_yticks([])

    fig.suptitle("Threshold strategy comparison on the 2026-04-01 CCOR-1 CME sequence", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = HERE / "ccor1_threshold_compare.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
