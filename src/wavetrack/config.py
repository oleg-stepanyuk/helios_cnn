"""Single-file configuration for the Wavetrack pipeline.

The configuration is loaded from a TOML file. A complete annotated example
is provided in ``examples/config.toml``.

Sections
--------
[io]            Input/output paths and formats.
[preprocess]    Clipping, inversion, normalization, ROI/disk masks.
[wavelet]       À-trous decomposition parameters: number of scales, kernel,
                per-scale weights for reconstruction.
[threshold]     Relative (sigma-based) thresholding per wavelet coefficient,
                with optional disk/limb split.
[segmentation]  Object/structure extraction parameters: gradient method,
                min size, reference scale, window filter.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Tuple

if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[import]
else:
    import tomli as tomllib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IOConfig:
    """Input / output settings."""

    # Path to the single input image (FITS / .npy / PNG / JPG). When provided
    # in the config it can be overridden via CLI argument.
    input_path: Optional[str] = None

    # Directory where outputs are written. Subdirectories are created
    # automatically.
    output_dir: str = "./wavetrack_output"

    # When loading FITS, which HDU index to read.
    fits_hdu_index: int = 0

    # Optional resize of the input image. ``None`` keeps the original shape.
    resize_to: Optional[Tuple[int, int]] = None

    # Save intermediate wavelet coefficients and scales as PNGs.
    save_intermediate: bool = True

    # Save final masks as FITS in addition to PNG.
    save_fits: bool = False


@dataclass
class PreprocessConfig:
    """Image preprocessing."""

    # Clip raw pixel values to this interval before any further processing.
    # Use very wide defaults to effectively disable clipping.
    low_threshold: float = -1.0e9
    high_threshold: float = 1.0e9

    # Invert the image after clipping. Useful for filaments observed on
    # bright disks (they appear as dark features in H-alpha / EUV).
    invert: bool = False

    # 'pil' uses ``PIL.ImageOps.invert`` after rescaling to 8-bit,
    # 'numeric' performs a simple ``max - x`` inversion preserving range.
    invert_method: str = "numeric"

    # Optional logarithmic dynamic-range compression (base e if base<=1).
    log_compress: bool = False
    log_base: float = 20.0

    # Normalize to N-bit resolution after preprocessing (posterization).
    # Set to 0 to skip.
    resolution_bits: int = 8

    # Optional circular mask (solar disk). When enabled, the disk is used to
    # split statistics between on-disk and off-limb regions in the wavelet
    # thresholding step.
    apply_circle_mask: bool = False
    disk_center_x: Optional[int] = None  # default: image center
    disk_center_y: Optional[int] = None
    disk_radius: Optional[int] = None  # required if ``apply_circle_mask``
    rsun_coeff: float = 1.0  # optional radius multiplier

    # Optional rectangular ROI mask. Pixels outside the box are zeroed.
    apply_square_mask: bool = False
    square_x1: int = 0
    square_x2: int = 0
    square_y1: int = 0
    square_y2: int = 0


@dataclass
class WaveletConfig:
    """À-trous wavelet decomposition."""

    # Number of decomposition levels J. Yields J wavelet coefficients
    # ``c_0..c_{J-1}`` and J-1 scales ``omega_j = c_{j-1} - c_j``.
    n_scales: int = 5

    # Per-scale weights used when reconstructing the image from scales.
    # The list length must be >= ``n_scales``. Extra entries are ignored.
    # Index j corresponds to scale omega_j. A typical CME-shock setup uses
    # ``[0, 1, 1, 0, 0]`` (highlight mid-frequency features).
    scales_weights: List[float] = field(
        default_factory=lambda: [0.0, 1.0, 1.0, 0.0, 0.0]
    )

    # Reference scale index applied as a binary mask to the reconstructed
    # image (limits objects to the spatial support of a coarse wavelet
    # coefficient).
    reference_scale: int = 1


@dataclass
class ThresholdConfig:
    """Relative (sigma-based) thresholding of wavelet coefficients."""

    # Apply threshold to each wavelet coefficient using its fitted Gaussian
    # standard deviation. If ``split_disk_limb`` is true, on-disk and
    # off-limb statistics are computed separately and the two halves merged.
    split_disk_limb: bool = False

    # Threshold in sigma units for the on-disk region (or whole image when
    # ``split_disk_limb`` is false).
    sigma_disk: float = 5.5

    # Threshold in sigma units for the off-limb region.
    sigma_limb: float = 5.5

    # Threshold applied to the gradient field.
    sigma_grad: float = 5.5


@dataclass
class SegmentationConfig:
    """Object/structure segmentation and post-processing."""

    # Gradient operator used as an additional structural criterion.
    # One of: 'sobel-feldman', 'simple', 'hagenaer', 'none'.
    grad_method: str = "sobel-feldman"

    # Minimum structure size in pixels. Smaller connected components are
    # discarded.
    min_struct_size: int = 400

    # Final one-pass window filter that suppresses small spurious features
    # at the edges of detected masks.
    window_filt_boxsize: int = 10
    window_filt_increment: int = 1
    window_filt_threshold: float = 0.0


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Top-level Wavetrack configuration."""

    io: IOConfig = field(default_factory=IOConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)

    # --- loading helpers ---------------------------------------------------

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        """Load configuration from a TOML file.

        Unknown keys raise a ``ValueError`` so typos are caught early.
        Missing sections fall back to defaults.
        """
        with open(path, "rb") as handle:
            data = tomllib.load(handle)

        cfg = cls()
        section_map = {
            "io": (cfg.io, IOConfig),
            "preprocess": (cfg.preprocess, PreprocessConfig),
            "wavelet": (cfg.wavelet, WaveletConfig),
            "threshold": (cfg.threshold, ThresholdConfig),
            "segmentation": (cfg.segmentation, SegmentationConfig),
        }

        for name, (current, dtype) in section_map.items():
            if name not in data:
                continue
            raw = data[name]
            if not isinstance(raw, dict):
                raise ValueError(f"Section [{name}] must be a table")
            unknown = set(raw) - {f.name for f in dtype.__dataclass_fields__.values()}
            if unknown:
                raise ValueError(
                    f"Unknown keys in [{name}]: {sorted(unknown)}"
                )
            # tomllib returns lists/tuples as lists. Normalize resize_to.
            if name == "io" and "resize_to" in raw and raw["resize_to"] is not None:
                raw["resize_to"] = tuple(raw["resize_to"])
            new_section = replace(current, **raw)
            setattr(cfg, name, new_section)

        cfg.validate()
        return cfg

    # --- validation --------------------------------------------------------

    def validate(self) -> None:
        """Sanity-check fields that have cross-section dependencies."""
        w = self.wavelet
        if w.n_scales < 2:
            raise ValueError("wavelet.n_scales must be >= 2")
        if len(w.scales_weights) < w.n_scales:
            raise ValueError(
                "wavelet.scales_weights must have at least n_scales entries"
            )
        if not (0 <= w.reference_scale < w.n_scales):
            raise ValueError("wavelet.reference_scale must be in [0, n_scales)")

        p = self.preprocess
        if p.apply_circle_mask and p.disk_radius is None:
            raise ValueError(
                "preprocess.disk_radius is required when apply_circle_mask=true"
            )
        if p.invert_method not in ("numeric", "pil"):
            raise ValueError("preprocess.invert_method must be 'numeric' or 'pil'")
        if p.resolution_bits < 0 or p.resolution_bits > 32:
            raise ValueError("preprocess.resolution_bits must be in [0, 32]")

        s = self.segmentation
        if s.grad_method not in ("sobel-feldman", "simple", "hagenaer", "none"):
            raise ValueError(
                "segmentation.grad_method must be one of: "
                "'sobel-feldman', 'simple', 'hagenaer', 'none'"
            )
