#!/usr/bin/env python3
"""Validate parabolic peak interpolation on a real STFT frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sms import DEFAULT_SMS_PARAMS, compute_stft_magnitude, load_mono_audio, parabolic_interpolation


def main() -> None:
    parser = argparse.ArgumentParser(description="Test parabolic peak interpolation.")
    parser.add_argument("input_wav", type=Path, help="Mono WAV file")
    parser.add_argument("--time", type=float, default=0.5, help="Frame time in seconds")
    args = parser.parse_args()

    if not args.input_wav.is_file():
        raise SystemExit(f"Input file not found: {args.input_wav}")

    params = DEFAULT_SMS_PARAMS
    data, sample_rate = load_mono_audio(str(args.input_wav))
    mag, times, freqs = compute_stft_magnitude(data, sample_rate, params.n_fft, params.hop_size)

    frame_idx = int(np.argmin(np.abs(times - args.time)))
    frame_mag = mag[frame_idx]

    valid_bins = np.where((freqs > 100) & (freqs < 500))[0]
    peak_bin = int(valid_bins[np.argmax(frame_mag[valid_bins])])

    print(f"Frame index: {frame_idx} (t = {times[frame_idx]:.3f} s)")
    print(f"Peak bin: {peak_bin}, freq: {freqs[peak_bin]:.2f} Hz, mag: {frame_mag[peak_bin]:.4f}")

    alpha = frame_mag[peak_bin - 1]
    beta = frame_mag[peak_bin]
    gamma = frame_mag[peak_bin + 1]
    print(f"alpha (k-1): {alpha:.4f}, beta (k): {beta:.4f}, gamma (k+1): {gamma:.4f}")

    result = parabolic_interpolation(frame_mag, peak_bin)
    if result is None:
        raise SystemExit("Parabolic interpolation failed for selected peak.")
    print(
        f"Refined bin: {result.bin_index:.3f}, "
        f"refined freq: {result.bin_index * freqs[1]:.2f} Hz, "
        f"refined amp: {result.amplitude:.4f}"
    )


if __name__ == "__main__":
    main()
