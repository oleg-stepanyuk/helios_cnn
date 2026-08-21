"""Parameter sweep for Wavetrack on the 2026-04-01 CCOR-1 CME sequence.

We sweep over a small grid of "knobs" likely to matter for difference-image
coronagraph data:

* ``threshold_mode``       : how to set [low, high] clipping bounds
    - "fixed_signed"       : symmetric ``[-T, T]`` with T = 5e-9
    - "p_lo_p_hi"          : per-frame percentiles (1%, 99.5%)
    - "p_zero_p_hi"        : per-frame ``[0, 99.5%]`` (positive-only)
* ``scales_profile``       : which à-trous scales to combine (binary weights)
    - "23"                 : ``[0,1,1,0,0]`` reference=1 — small/medium features
    - "234"                : ``[0,1,1,1,0]`` reference=2 — adds large features
    - "34"                 : ``[0,0,1,1,0]`` reference=2 — medium-large only
* ``sigma``                : detection threshold in σ units (3.0 vs 3.5 vs 4.5)
* ``min_struct_size``      : minimum component size (100 vs 300 vs 600)

For each (frame, config) we run Wavetrack and record:
* number of detected structures
* total mask pixel area
* largest-component fraction (to detect blow-up / runaway detections)
* center-of-mass of biggest component (sanity check vs known CME direction NW)

Then we pick the config that:
  (a) detects >=1 structure on every "active" frame (>=00:30 UT),
  (b) detects 0 structures (or very few small ones) on the pre-event baseline (23:00 UT diff = 23:45 - 23:00 if any),
  (c) does not blow up to >50% of the image,
  (d) gives the largest combined mask area on the peak frame (04:00 UT).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from astropy.io import fits

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*BLANK keyword.*")

sys.path.insert(0, "/home/user/workspace/wavetrack/src")
from wavetrack import Config, run  # noqa: E402

HERE = Path(__file__).resolve().parent
DIFFS = HERE / "diffs"
SWEEP_DIR = HERE / "sweep_out"
SWEEP_DIR.mkdir(exist_ok=True)

# Use base-diffs against 23:00 UT — these accumulate the CME signal cleanly.
FRAMES = [
    ("23:45", DIFFS / "ccor1_basediff_2345_minus_2300.fits"),
    ("00:30", DIFFS / "ccor1_basediff_0030_minus_2300.fits"),
    ("01:30", DIFFS / "ccor1_basediff_0130_minus_2300.fits"),
    ("03:00", DIFFS / "ccor1_basediff_0300_minus_2300.fits"),
    ("04:00", DIFFS / "ccor1_basediff_0400_minus_2300.fits"),
]


@dataclass
class FrameStats:
    p1: float
    p99: float
    p995: float
    p999: float
    median: float


def frame_stats(path: Path) -> FrameStats:
    with fits.open(path) as hdul:
        d = np.asarray(hdul[0].data, dtype=np.float32)
    d = d[np.isfinite(d)]
    return FrameStats(
        p1=float(np.percentile(d, 1)),
        p99=float(np.percentile(d, 99)),
        p995=float(np.percentile(d, 99.5)),
        p999=float(np.percentile(d, 99.9)),
        median=float(np.median(d)),
    )


THRESHOLD_MODES = ["fixed_signed", "p_lo_p_hi", "p_zero_p_hi"]
SCALE_PROFILES = {
    "23":  ([0.0, 1.0, 1.0, 0.0, 0.0], 1),
    "234": ([0.0, 1.0, 1.0, 1.0, 0.0], 2),
    "34":  ([0.0, 0.0, 1.0, 1.0, 0.0], 2),
}
SIGMAS = [3.0, 4.0]
MIN_SIZES = [200, 500]


def make_cfg(
    input_path: Path,
    out_dir: Path,
    threshold_mode: str,
    stats: FrameStats,
    scale_key: str,
    sigma: float,
    min_size: int,
) -> Config:
    cfg = Config()
    cfg.io.input_path = str(input_path)
    cfg.io.fits_hdu_index = 0
    cfg.io.output_dir = str(out_dir)
    cfg.io.save_intermediate = False  # save speed/space
    cfg.io.save_fits = False

    # threshold mode
    if threshold_mode == "fixed_signed":
        cfg.preprocess.low_threshold = -5.0e-9
        cfg.preprocess.high_threshold = 5.0e-9
    elif threshold_mode == "p_lo_p_hi":
        cfg.preprocess.low_threshold = stats.p1
        cfg.preprocess.high_threshold = stats.p995
    elif threshold_mode == "p_zero_p_hi":
        cfg.preprocess.low_threshold = 0.0
        cfg.preprocess.high_threshold = stats.p995
    else:
        raise ValueError(threshold_mode)

    cfg.preprocess.invert = False
    cfg.preprocess.resolution_bits = 0  # critical: no posterization on floats
    cfg.preprocess.apply_circle_mask = False  # occulter is already blanked

    weights, ref = SCALE_PROFILES[scale_key]
    cfg.wavelet.n_scales = 5
    cfg.wavelet.scales_weights = weights
    cfg.wavelet.reference_scale = ref

    cfg.threshold.split_disk_limb = False
    cfg.threshold.sigma_disk = sigma
    cfg.threshold.sigma_limb = sigma
    cfg.threshold.sigma_grad = sigma

    cfg.segmentation.min_struct_size = min_size
    return cfg


def score_result(seg) -> Dict:
    n = seg.n_structures
    sizes = sorted([s.size for s in seg.structures], reverse=True)
    total = int(sum(sizes))
    largest = int(sizes[0]) if sizes else 0
    coms = [s.center_of_mass for s in seg.structures]
    return {
        "n_structures": int(n),
        "total_mask_pixels": total,
        "largest_pixels": largest,
        "sizes_top5": sizes[:5],
        "coms_top5": [[float(c[0]), float(c[1])] for c in coms[:5]],
    }


def main() -> None:
    print("==> per-frame stats:")
    stats = {label: frame_stats(p) for label, p in FRAMES}
    for label, s in stats.items():
        print(f"  {label}: median={s.median:.3e}, p99.5={s.p995:.3e}, p99.9={s.p999:.3e}")

    image_area = 1920 * 2048

    results: List[Dict] = []
    configs = [
        (tm, sk, sg, ms)
        for tm in THRESHOLD_MODES
        for sk in SCALE_PROFILES
        for sg in SIGMAS
        for ms in MIN_SIZES
    ]
    print(f"\n==> sweeping {len(configs)} configs across {len(FRAMES)} frames "
          f"= {len(configs) * len(FRAMES)} runs")
    t0 = time.time()
    for ci, (tm, sk, sg, ms) in enumerate(configs):
        cfg_label = f"thr={tm}_sc={sk}_sig={sg}_min={ms}"
        per_frame = []
        for fl, fp in FRAMES:
            out_dir = SWEEP_DIR / cfg_label / fl.replace(":", "")
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg = make_cfg(fp, out_dir, tm, stats[fl], sk, sg, ms)
            try:
                res = run(input_path=str(fp), config=cfg)
                m = score_result(res.segmentation)
                # cleanup heavy artifacts — we only need numbers for the sweep
                shutil.rmtree(out_dir, ignore_errors=True)
            except Exception as exc:  # noqa: BLE001
                m = {"error": str(exc)}
            m["frame"] = fl
            per_frame.append(m)
        results.append({"config": cfg_label, "params": {"threshold_mode": tm, "scales": sk, "sigma": sg, "min_size": ms},
                        "per_frame": per_frame})
        dt = time.time() - t0
        eta = dt / (ci + 1) * (len(configs) - ci - 1)
        print(f"  [{ci+1:>3}/{len(configs)}] {cfg_label}  (elapsed {dt:.0f}s, eta {eta:.0f}s)")

    # scoring rubric
    def evaluate(cfg_entry: Dict) -> Tuple[float, Dict]:
        pf = cfg_entry["per_frame"]
        # map by frame
        pfd = {p["frame"]: p for p in pf}
        peak = pfd["04:00"]
        early = pfd["00:30"]
        mid = pfd["01:30"]
        late = pfd["03:00"]
        pre = pfd["23:45"]

        diagnostics = {}
        # disqualifiers
        for f in ("00:30", "01:30", "03:00", "04:00"):
            if "error" in pfd[f]:
                return -1e9, {"reason": f"error on {f}: {pfd[f]['error']}"}
            if pfd[f]["largest_pixels"] / image_area > 0.40:
                diagnostics[f"blowup_{f}"] = pfd[f]["largest_pixels"] / image_area
                return -1e6, {"reason": f"blowup on {f} ({pfd[f]['largest_pixels']/image_area:.2%})"}

        # main reward: peak total mask, growth from early to peak
        peak_total = peak["total_mask_pixels"]
        peak_n = peak["n_structures"]
        early_total = early["total_mask_pixels"]
        growth = peak_total - early_total

        # penalty: detections on pre-event (23:45) when the CME isn't there yet
        pre_total = pre["total_mask_pixels"]
        pre_penalty = pre_total  # 1:1 penalty

        # bonus: monotonic growth early -> mid -> late -> peak
        monotonic_bonus = 0
        seq = [early_total, mid["total_mask_pixels"], late["total_mask_pixels"], peak_total]
        if all(b >= a * 0.7 for a, b in zip(seq, seq[1:])):  # allow modest dips
            monotonic_bonus = 5000

        # penalty for huge spurious detections
        # already filtered out blow-ups; penalize fragmentation (n_structures over 20)
        frag_penalty = max(0, peak_n - 20) * 200

        score = peak_total + 0.5 * growth - pre_penalty - frag_penalty + monotonic_bonus
        diagnostics.update({
            "peak_total": peak_total, "peak_n": peak_n,
            "early_total": early_total, "growth": growth,
            "pre_total_2345": pre_total, "frag_penalty": frag_penalty,
            "monotonic_bonus": monotonic_bonus,
        })
        return score, diagnostics

    scored = []
    for r in results:
        s, diag = evaluate(r)
        scored.append((s, r, diag))
    scored.sort(key=lambda t: t[0], reverse=True)

    print("\n==> Top 5 configs:")
    for rank, (s, r, diag) in enumerate(scored[:5], 1):
        print(f"  #{rank}  score={s:.0f}  {r['config']}")
        print(f"       diag={diag}")
        for p in r["per_frame"]:
            if "error" in p:
                print(f"        {p['frame']}: ERROR {p['error']}")
            else:
                print(f"        {p['frame']}: n={p['n_structures']:>3}  total={p['total_mask_pixels']:>7}  largest={p['largest_pixels']:>7}  sizes={p['sizes_top5']}")

    # save full results
    (SWEEP_DIR / "sweep_results.json").write_text(json.dumps([
        {"score": s, "config": r["config"], "params": r["params"], "diag": diag, "per_frame": r["per_frame"]}
        for s, r, diag in scored
    ], indent=2))
    print(f"\n==> full sweep results written to {SWEEP_DIR / 'sweep_results.json'}")
    print(f"==> best config: {scored[0][1]['config']}")


if __name__ == "__main__":
    main()
