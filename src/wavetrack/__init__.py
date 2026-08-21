"""Wavetrack: à-trous wavelet decomposition and multi-scale object
recognition for solar eruptive features.

Reference:
    Stepanyuk, O., Kozarev, K., & Nedal, M. (2022).
    *Multi-scale image preprocessing and feature tracking for remote
    CME characterization*. J. Space Weather Space Clim. 12, 20.
    https://doi.org/10.1051/swsc/2022020
"""

from .config import (
    Config,
    IOConfig,
    PreprocessConfig,
    SegmentationConfig,
    ThresholdConfig,
    WaveletConfig,
)
from .pipeline import PipelineResult, run
from .wavelet import A_TROUS_KERNEL, WaveletDecomposition, decompose, dilate_kernel

__all__ = [
    "Config",
    "IOConfig",
    "PreprocessConfig",
    "SegmentationConfig",
    "ThresholdConfig",
    "WaveletConfig",
    "PipelineResult",
    "run",
    "A_TROUS_KERNEL",
    "WaveletDecomposition",
    "decompose",
    "dilate_kernel",
]

__version__ = "1.0.0"
