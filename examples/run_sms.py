#!/usr/bin/env python3
"""
Run Spectral Modeling Synthesis on a mono WAV file.

Usage:
    python examples/run_sms.py path/to/input.wav
    python examples/run_sms.py path/to/input.wav --output-dir output/
    python examples/run_sms.py path/to/input.wav --config config/sms_defaults.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sms import DEFAULT_SMS_PARAMS, export_wav, load_config, load_mono_audio, spectral_modeling_synthesis


def _stem(path: Path) -> str:
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose a mono WAV into deterministic and stochastic SMS branches."
    )
    parser.add_argument("input_wav", type=Path, help="Input mono WAV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output",
        help="Directory for output WAV files (default: output/)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config path (default: config/sms_defaults.yaml)",
    )
    args = parser.parse_args()

    if not args.input_wav.is_file():
        raise SystemExit(f"Input file not found: {args.input_wav}")

    params = load_config(args.config) if args.config else DEFAULT_SMS_PARAMS
    original, sample_rate_hz = load_mono_audio(str(args.input_wav))

    print("Spectral Modeling Synthesis (Serra & Smith, 1990)")
    print(f"  Input: {args.input_wav}")
    print(f"  Sample rate: {sample_rate_hz} Hz")
    print(f"  Duration: {len(original) / sample_rate_hz:.3f} s")
    print(
        f"  Analysis: {params.analysis_window}, N={params.n_fft}, H={params.hop_size}, "
        f"Δf={params.frequency_resolution_hz:.2f} Hz"
    )

    rng = np.random.default_rng(params.random_seed)
    det, stoch, full, analysis = spectral_modeling_synthesis(
        original, sample_rate_hz, rng=rng, params=params
    )

    print(f"  Estimated f0: {analysis.f0_hz:.2f} Hz")
    print(f"  Tracked partials: {len(analysis.partials)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(args.input_wav)
    det_path = args.output_dir / f"{stem}_deterministic.wav"
    stoch_path = args.output_dir / f"{stem}_stochastic.wav"
    full_path = args.output_dir / f"{stem}_resynthesis.wav"

    export_wav(str(det_path), det, sample_rate_hz)
    export_wav(str(stoch_path), stoch, sample_rate_hz)
    export_wav(str(full_path), full, sample_rate_hz)

    print(f"  Wrote: {det_path}")
    print(f"  Wrote: {stoch_path}")
    print(f"  Wrote: {full_path}")


if __name__ == "__main__":
    main()
