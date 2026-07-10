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

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sms import DEFAULT_SMS_PARAMS, load_config, load_mono_audio, spectral_modeling_synthesis


def _narrowband_spectrogram(
    x: np.ndarray,
    sample_rate_hz: int,
    n_fft: int = 2048,
    hop_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = signal.windows.blackmanharris(n_fft, sym=True)
    freqs_hz, times_s, spec = signal.spectrogram(
        x,
        fs=sample_rate_hz,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_size,
        mode="magnitude",
        scaling="spectrum",
    )
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))
    return freqs_hz, times_s, spec_db


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

    branches = [
        ("Original", original),
        ("Deterministic", det),
        ("Stochastic", stoch),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (title, sig) in zip(axes, branches, strict=True):
        freqs_hz, times_s, spec_db = _narrowband_spectrogram(sig, sample_rate_hz)
        mask = freqs_hz <= args.fmax
        mesh = ax.pcolormesh(
            times_s,
            freqs_hz[mask] / 1000.0,
            spec_db[mask, :],
            shading="gouraud",
            cmap="magma",
            vmin=-80,
            vmax=0,
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")

    fig.colorbar(mesh, ax=axes, label="Magnitude (dB)", fraction=0.02, pad=0.02)
    fig.suptitle(
        f"SMS decomposition — f0 ≈ {analysis.f0_hz:.1f} Hz, k = {len(analysis.partials)} partials",
        fontsize=11,
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
