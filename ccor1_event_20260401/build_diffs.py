"""Build CCOR-1 difference frames for the 2026-04-01 CME sequence.

Each diff is post - base, where 'base' is the pre-event reference (23:00 UT).
We also build adjacent diffs (frame N - frame N-1) which highlight bright leading-edge motion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent
FRAMES = [
    ("23:00", "CCOR1_2_20260401T230025_V00_NC.fits"),
    ("23:45", "CCOR1_2_20260401T234525_V00_NC.fits"),
    ("00:30", "CCOR1_2_20260402T003025_V00_NC.fits"),
    ("01:30", "CCOR1_2_20260402T013025_V00_NC.fits"),
    ("03:00", "CCOR1_2_20260402T030025_V00_NC.fits"),
    ("04:00", "CCOR1_2_20260402T040025_V00_NC.fits"),
]


def load(p: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(p) as hdul:
        data = np.asarray(hdul[1].data, dtype=np.float32)
        hdr = hdul[1].header.copy()
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, hdr


def write_single_hdu(out: Path, data: np.ndarray, ref_header: fits.Header, comment: str) -> None:
    hdu = fits.PrimaryHDU(data=data)
    for key in ("DATE-OBS", "TELESCOP", "INSTRUME", "WAVELNTH", "BUNIT"):
        if key in ref_header:
            hdu.header[key] = ref_header[key]
    hdu.header["HISTORY"] = comment
    hdu.writeto(out, overwrite=True)


def main() -> None:
    base_data, base_hdr = load(HERE / FRAMES[0][1])
    prev_data, prev_hdr = base_data, base_hdr

    out_dir = HERE / "diffs"
    out_dir.mkdir(exist_ok=True)

    # base-diffs against 23:00 UT
    for label, name in FRAMES[1:]:
        data, hdr = load(HERE / name)
        diff = data - base_data
        tag = label.replace(":", "")
        write_single_hdu(out_dir / f"ccor1_basediff_{tag}_minus_2300.fits", diff, hdr, f"diff = {label} - 23:00 UT (base)")
        prev_data, prev_hdr = data, hdr

    # running diffs (N - N-1)
    prev_data, prev_hdr = base_data, base_hdr
    for label, name in FRAMES[1:]:
        data, hdr = load(HERE / name)
        diff = data - prev_data
        tag = label.replace(":", "")
        write_single_hdu(out_dir / f"ccor1_rundiff_{tag}.fits", diff, hdr, f"running diff at {label}")
        prev_data, prev_hdr = data, hdr

    print(f"Wrote diff files to {out_dir}")
    for f in sorted(out_dir.glob("*.fits")):
        print(" ", f.name)


if __name__ == "__main__":
    main()
