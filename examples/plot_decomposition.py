#!/usr/bin/env python3
"""
Plot a three-panel decomposition spectrogram (original / deterministic / stochastic).

Usage:
    python examples/plot_decomposition.py path/to/input.wav
    python examples/plot_decomposition.py path/to/input.wav --output output/decomposition.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.sms_visuals import save_decomposition_figure
from sms import DEFAULT_SMS_PARAMS, load_config, load_mono_audio, spectral_modeling_synthesis


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SMS decomposition spectrograms.")
    parser.add_argument("input_wav", type=Path, help="Input mono WAV file")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "decomposition.png",
        help="Output image path (default: output/decomposition.png)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path")
    parser.add_argument("--fmax", type=float, default=8000.0, help="Max frequency axis (Hz)")
    args = parser.parse_args()

    if not args.input_wav.is_file():
        raise SystemExit(f"Input file not found: {args.input_wav}")

    params = load_config(args.config) if args.config else DEFAULT_SMS_PARAMS
    original, sample_rate_hz = load_mono_audio(str(args.input_wav))
    det, stoch, _, analysis = spectral_modeling_synthesis(
        original,
        sample_rate_hz,
        params=params,
    )

    path = save_decomposition_figure(
        original,
        det,
        stoch,
        sample_rate_hz,
        analysis,
        args.output,
        fmax_hz=args.fmax,
        n_fft=params.n_fft,
        hop_size=params.hop_size,
    )
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
