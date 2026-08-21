"""Command-line entry point for Wavetrack."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .pipeline import run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wavetrack",
        description=(
            "A-trous wavelet image decomposition and multi-scale object "
            "recognition for solar eruptive features."
        ),
    )
    parser.add_argument(
        "--image",
        "-i",
        type=Path,
        required=False,
        help="Path to the input image (FITS / .npy / PNG / JPG). "
        "If omitted, io.input_path from the config is used.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to the TOML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Override the output directory from the config.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = Config.from_toml(args.config)
    if args.output_dir is not None:
        cfg.io.output_dir = str(args.output_dir)

    image_path = str(args.image) if args.image else None
    result = run(input_path=image_path, config=cfg)
    print(
        f"Done. Detected {result.n_structures} structures. "
        f"Outputs in: {result.output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
