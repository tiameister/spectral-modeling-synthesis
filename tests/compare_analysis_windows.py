#!/usr/bin/env python3
"""
Compare peak-continuation partial trajectories under different STFT analysis windows.

Usage:
    python tests/compare_analysis_windows.py --input path/to/mono.wav
    python tests/compare_analysis_windows.py --input path/to/mono.wav --windows hann kaiser --kaiser-betas 6 8 10
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sms import DEFAULT_SMS_PARAMS, extract_sinusoidal_partials, load_mono_audio

TEST_DIR = Path(__file__).resolve().parent
TRAJECTORY_T_MAX_S = 1.5
F_CHAOS_HZ = 2000.0
F_MAX_HZ = 8000.0

DEFAULT_WINDOWS: list[tuple[str, float | None]] = [
    ("blackman_harris", None),
    ("hann", None),
    ("hamming", None),
    ("blackman", None),
    ("kaiser", 4.0),
    ("kaiser", 6.0),
    ("kaiser", 8.0),
    ("kaiser", 10.0),
    ("kaiser", 12.0),
    ("kaiser", 14.0),
]


def _narrowband_spectrogram(x: np.ndarray, sample_rate_hz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy import signal

    n_fft = DEFAULT_SMS_PARAMS.n_fft
    hop_size = DEFAULT_SMS_PARAMS.hop_size
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


def _slug(window_name: str, beta: float | None) -> str:
    if window_name == "kaiser" and beta is not None:
        beta_str = str(beta).replace(".", "p")
        return f"kaiser_beta{beta_str}"
    return re.sub(r"[^a-z0-9]+", "_", window_name.lower()).strip("_")


def _high_band_partial_count(analysis, f_min_hz: float = F_CHAOS_HZ) -> int:
    count = 0
    for partial in analysis.partials:
        if float(np.max(partial.frequencies_hz)) >= f_min_hz:
            count += 1
    return count


def plot_partial_trajectories_window(
    original: np.ndarray,
    analysis,
    sample_rate_hz: int,
    window_label: str,
    output_path: Path,
    t_max_s: float = TRAJECTORY_T_MAX_S,
) -> None:
    freqs_hz, times_s, spec_db = _narrowband_spectrogram(original, sample_rate_hz)
    freq_mask = freqs_hz <= F_MAX_HZ
    t_cap = min(t_max_s, times_s[-1])
    t_lo = float(times_s[0])
    f_plot_khz = freqs_hz[freq_mask] / 1000.0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.pcolormesh(
        times_s,
        f_plot_khz,
        spec_db[freq_mask, :],
        shading="nearest",
        cmap="magma",
        vmin=-80,
        vmax=0,
    )
    for partial in analysis.partials:
        mask = (partial.times_s >= t_lo) & (partial.times_s <= t_cap + 1e-9)
        if not np.any(mask):
            continue
        ax.plot(partial.times_s[mask], partial.frequencies_hz[mask] / 1000.0, color="#ED8C01", linewidth=1.5)

    ax.set_xlim(t_lo, t_cap)
    ax.set_ylim(0.0, min(F_MAX_HZ, 3500.0) / 1000.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(f"Analysis window: {window_label} — k = {len(analysis.partials)} partials")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_comparison(
    input_wav: str,
    window_configs: list[tuple[str, float | None]],
) -> None:
    original, sr = load_mono_audio(input_wav)
    out_dir = TEST_DIR / "window_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "SMS analysis-window comparison (peak continuation only)",
        f"Input: {input_wav}",
        f"N_FFT={DEFAULT_SMS_PARAMS.n_fft}, H={DEFAULT_SMS_PARAMS.hop_size}",
        "",
        f"{'Window':<28} {'f0 (Hz)':>10} {'k total':>10} {'k >2kHz':>10}",
        "-" * 62,
    ]

    print(f"Window comparison on {input_wav}")
    for window_name, beta in window_configs:
        params = replace(
            DEFAULT_SMS_PARAMS,
            analysis_window=window_name,
            analysis_window_beta=beta,
        )
        label = params.analysis_window_display_name
        slug = _slug(window_name, beta)
        print(f"  [{label}] ...", flush=True)

        analysis = extract_sinusoidal_partials(original, sr, params)
        n_high = _high_band_partial_count(analysis)
        out_pdf = out_dir / f"plot_trajectories_{slug}.png"
        plot_partial_trajectories_window(original, analysis, sr, label, out_pdf)

        summary_lines.append(
            f"{label:<28} {analysis.f0_hz:10.2f} {len(analysis.partials):10d} {n_high:10d}"
        )
        print(f"    k={len(analysis.partials)}, k(>{F_CHAOS_HZ} Hz)={n_high}, f0={analysis.f0_hz:.2f} Hz")
        print(f"    -> {out_pdf.name}")

    summary_path = out_dir / "window_comparison_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nSummary: {summary_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SMS trajectories across analysis windows.")
    parser.add_argument("--input", required=True, help="Mono WAV path")
    parser.add_argument("--windows", nargs="+", default=None, help="Window names (default: full sweep)")
    parser.add_argument(
        "--kaiser-betas",
        nargs="+",
        type=float,
        default=[4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
        help="Kaiser beta values when 'kaiser' is in --windows",
    )
    return parser.parse_args()


def _build_window_configs(args: argparse.Namespace) -> list[tuple[str, float | None]]:
    if args.windows is None:
        return DEFAULT_WINDOWS

    configs: list[tuple[str, float | None]] = []
    for name in args.windows:
        key = name.strip().lower()
        if key == "kaiser":
            for beta in args.kaiser_betas:
                configs.append(("kaiser", float(beta)))
        else:
            configs.append((key, None))
    return configs


def main() -> None:
    args = _parse_args()
    run_comparison(args.input, _build_window_configs(args))


if __name__ == "__main__":
    main()
