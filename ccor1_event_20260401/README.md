# Wavetrack — CCOR-1 CME event sweep (2026-04-01)

This bundle runs Wavetrack on a short CCOR-1 (GOES-19) sequence of a
well-documented Earth-directed CME and reports the best parameter set.

## Event

* **Date**: 2026-04-01 23:45 UT (onset) — 2026-04-02 04:00+ UT (peak)
* **Type**: Large partial halo CME, NW-directed
* **Source**: filament eruption near N28W25, associated with a long-duration
  C-class flare and a strong filament-channel destabilization beginning
  ~22:30 UT on 2026-04-01.
* **Geo-effect**: Max Kp 6.67 (per CCMC CME Scoreboard).

## Frames used

| Time (UT) | Role |
|---|---|
| 23:00, 2026-04-01 | pre-event baseline (subtracted from every other frame) |
| 23:45, 2026-04-01 | onset (CME just lifting off) |
| 00:30, 2026-04-02 | early CME bulk on east limb |
| 01:30, 2026-04-02 | mid-expansion, CME envelope visible |
| 03:00, 2026-04-02 | late-bright, CME crossing FOV |
| 04:00, 2026-04-02 | peak brightness |

Inputs are **base-difference frames**: `frame(t) − frame(23:00 UT)`, single
primary HDU, float32, MSB units. Built by `build_diffs.py`.

## Parameter sweep

`sweep.py` runs Wavetrack on each frame across a grid of:

| Knob | Values |
|---|---|
| `threshold_mode` | `fixed_signed` (`±5e-9`), `p_lo_p_hi` (per-frame [p1, p99.5]), `p_zero_p_hi` ([0, p99.5]) |
| `scales_weights` | `[0,1,1,0,0]` ref=1, `[0,1,1,1,0]` ref=2, `[0,0,1,1,0]` ref=2 |
| `sigma` (disk/limb/grad) | 3.0, 4.0 |
| `min_struct_size` | 200, 500 |

36 configs × 5 frames = 180 runs.

The scoring rubric rewarded: (a) large total mask area on the peak frame,
(b) growth from early to peak, (c) small pre-event area at 23:45 UT;
penalized fragmentation and disqualified runs that blew up beyond 40% of
the image.

### Top config from automated scoring

`thr=fixed_signed_sc=23_sig=3.0_min=500` — score 102 092

But the side-by-side comparison (`ccor1_threshold_compare.png`) shows that
**percentile per-frame thresholds** capture the CME envelope much better at
all stages, even though the rubric penalized them for finding more
structure at the 23:45 onset frame. So the recommended config is:

| Parameter | Value | Why |
|---|---|---|
| `low_threshold` | per-frame **p1** | Clips negative outliers (saturation, ringing); adapts to frame range |
| `high_threshold` | per-frame **p99.5** | Same on the positive side; rejects the strongest 0.5% (cosmic rays, hot pixels) |
| `invert` | `false` | CME signal is positive in a difference frame |
| `resolution_bits` | `0` | **Critical**: posterization destroys the σ-fit on float coronagraph data |
| `apply_circle_mask` | `false` | Occulter is already blanked by L2 processing |
| `n_scales` | `5` | Standard à-trous depth |
| `scales_weights` | `[0, 1, 1, 0, 0]` | Combines scales 2 and 3 — features ~4–16 px wide |
| `reference_scale` | `1` | σ statistics from scale 1 |
| `split_disk_limb` | `false` | Coronagraph image has no solar disk |
| `sigma_disk`/`limb`/`grad` | `3.0` | Lower σ admits faint CME wings |
| `min_struct_size` | `500` | Drops noise blobs but keeps the CME envelope and its lobes |

## Results with the recommended config

`render_final.py` produces the 5×3 grid (`ccor1_event_grid.png`).

| Frame | n_structures | Top sizes (px) | Interpretation |
|---|---|---|---|
| 23:45 UT | 8 | 19 825 / 14 840 / 6 956 / 6 717 | pre-eruption coronal activity around the occulter |
| 00:30 UT | 3 | **76 855** / 1 665 / 638 | dominant CME bulk on the east limb |
| 01:30 UT | 13 | 41 690 / 19 960 / 14 235 / 8 992 | expanding CME envelope; multiple lobes |
| 03:00 UT | 7 | **44 818** / 3 181 / 2 619 | bright N-pylon plume + scattered debris |
| 04:00 UT | 3 | 21 651 / 20 127 / 20 011 | rim-saturated peak (CME fills FOV) |

## Threshold strategy comparison

See `ccor1_threshold_compare.png` for a 3×3 grid that holds the same
sigma/scales/min_size but varies only the threshold mode across the 00:30,
01:30 and 04:00 frames. Key takeaways:

* On the **early CME** (00:30): percentile captures 79k px vs 43k for fixed.
* On the **mid CME** (01:30): percentile captures 101k px / 13 components vs 17k px / 8 — the difference is dramatic.
* On the **peak** (04:00): all three converge (~60–91k px). When the CME is bright enough, threshold choice no longer matters.

## Performance note

The pure-Python `window_filter` cleanup loop in `segmentation.py` was a
~50 s bottleneck per frame. I JIT-compiled it with Numba; new runtime is
~4 s per frame (12× speed-up, identical numerical output). The Numba path
is now built into the rewritten Wavetrack package.

## Files

* `build_diffs.py` — produces the 10 difference frames in `diffs/`
* `sweep.py` — parameter sweep, writes `sweep_out/sweep_results.json`
* `render_final.py` — runs the recommended config end-to-end, makes `ccor1_event_grid.png`
* `render_compare.py` — produces `ccor1_threshold_compare.png`
* `final_out/<time>/` — per-frame Wavetrack outputs (masks, scales, structures.csv, visualize/)
* `ccor1_event_grid.png` — main result figure
* `ccor1_threshold_compare.png` — threshold-strategy comparison

## Caveats / honest limitations

1. The **23:45 UT frame** shows ~50k pixels of detected activity even though the bulk CME hasn't lifted off yet. This is a mix of true pre-eruption signal (filament-channel destabilization, see CCMC notes) and artifacts of the percentile threshold being adaptive to a quiet frame's noise.
2. The **04:00 UT peak** is so saturated that the algorithm tracks the *occulter rim and N-pylon* where the CME light dominates, rather than separating CME components.
3. The automated sweep score I used is a heuristic, not a ground-truth metric. The fixed-signed config "wins" the score but the percentile config wins the visual eye test for CME-front tracking — see the comparison figure.
4. The 01:30 frame fragments into 13 components — this is the expanding envelope, but Wavetrack as built does not link components across frames. For true CME tracking you'd want a frame-to-frame association step (Hungarian assignment or IoU-based linker).
