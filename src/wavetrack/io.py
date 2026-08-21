"""Image loaders and savers for FITS / NumPy / PNG / JPG inputs.

All loaders return a 2-D ``float64`` ``numpy.ndarray``. RGB inputs are
converted to grayscale using ITU-R BT.601 luma coefficients.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy import of astropy / imageio so the package can be imported without
# them (errors are raised only when actually loading/saving FITS).
try:
    from astropy.io import fits as _fits
except Exception:  # pragma: no cover
    _fits = None

try:
    import imageio.v3 as _iio
except Exception:  # pragma: no cover
    _iio = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FITS_EXTS = (".fits", ".fit", ".fts")
_NPY_EXTS = (".npy",)
_RASTER_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_image(
    path: str | Path,
    fits_hdu_index: int = 0,
    resize_to: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, dict]:
    """Load a 2-D image from FITS, NumPy or a raster file.

    Parameters
    ----------
    path
        Path to the input file. The format is inferred from the extension.
    fits_hdu_index
        Which HDU to read when the input is FITS.
    resize_to
        Optional ``(width, height)`` to resize the loaded image to.

    Returns
    -------
    image : ndarray
        Two-dimensional ``float64`` array.
    header : dict
        A dictionary of metadata. For FITS this contains the FITS header
        cards converted to a plain dict; for other formats it is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    ext = path.suffix.lower()

    if ext in _FITS_EXTS:
        image, header = _load_fits(path, hdu_index=fits_hdu_index)
    elif ext in _NPY_EXTS:
        image, header = _load_npy(path)
    elif ext in _RASTER_EXTS:
        image, header = _load_raster(path)
    else:
        raise ValueError(
            f"Unsupported input extension {ext!r}. "
            f"Supported: {_FITS_EXTS + _NPY_EXTS + _RASTER_EXTS}"
        )

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2-D image, got shape {image.shape} from {path}"
        )

    image = image.astype(np.float64, copy=False)

    # Replace NaN / +inf / -inf with 0 so downstream stats are well-defined.
    # Real-world FITS frames (CCOR, K-Cor, LASCO) commonly include masked
    # pixels stored as NaN; without this they propagate through every
    # subsequent operation and crash the wavelet σ fit.
    if not np.isfinite(image).all():
        n_bad = int((~np.isfinite(image)).sum())
        logger.warning(
            "Replacing %d non-finite pixels (NaN / inf) with 0 in %s",
            n_bad,
            path,
        )
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    if resize_to is not None:
        image = _resize(image, target_size=resize_to)

    logger.info(
        "Loaded %s: shape=%s dtype=%s min=%.3g max=%.3g",
        path,
        image.shape,
        image.dtype,
        np.min(image),
        np.max(image),
    )
    return image, header


def _load_fits(path: Path, hdu_index: int) -> Tuple[np.ndarray, dict]:
    if _fits is None:
        raise ImportError("astropy is required to load FITS files")
    with _fits.open(path) as hdul:
        data = hdul[hdu_index].data
        header = dict(hdul[hdu_index].header)
    if data is None:
        raise ValueError(f"FITS HDU {hdu_index} of {path} has no data")
    return np.asarray(data), header


def _load_npy(path: Path) -> Tuple[np.ndarray, dict]:
    data = np.load(path)
    return np.asarray(data), {}


def _load_raster(path: Path) -> Tuple[np.ndarray, dict]:
    if _iio is not None:
        arr = np.asarray(_iio.imread(path))
    else:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        # RGB(A) → grayscale via BT.601 luma
        rgb = arr[..., :3].astype(np.float64)
        arr = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return arr, {}


def _resize(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    width, height = target_size
    pil = Image.fromarray(image)
    pil = pil.resize((width, height), resample=Image.BICUBIC)
    return np.asarray(pil, dtype=np.float64)


# ---------------------------------------------------------------------------
# Savers
# ---------------------------------------------------------------------------


def save_png(image: np.ndarray, path: str | Path, cmap: str = "gray") -> None:
    """Save a 2-D image as PNG with min/max stretching.

    The image is rescaled to [0, 255] using its own min/max. Constant images
    are saved as black.
    """
    import matplotlib.pyplot as plt  # local import; optional at runtime

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        arr = np.zeros_like(arr)
    plt.imsave(str(path), arr, cmap=cmap, origin="lower")


def save_fits(
    image: np.ndarray,
    path: str | Path,
    header: Optional[dict] = None,
) -> None:
    """Save a 2-D image as a FITS file."""
    if _fits is None:
        raise ImportError("astropy is required to save FITS files")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = _fits.Header()
    if header:
        for key, value in header.items():
            try:
                hdr[key[:8]] = value  # FITS keyword length is 8
            except Exception:
                continue
    hdu = _fits.PrimaryHDU(data=np.asarray(image), header=hdr)
    hdu.writeto(path, overwrite=True)


def save_numpy(array: np.ndarray, path: str | Path) -> None:
    """Save an array as ``.npy``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array))


def ensure_dir(path: str | Path) -> Path:
    """Create the directory if missing and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
