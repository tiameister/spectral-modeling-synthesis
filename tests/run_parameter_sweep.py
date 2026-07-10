#!/usr/bin/env python3
"""
SMS parameter sweep — writes metrics and WAV outputs under tests/sweep/.

Usage:
    python tests/run_parameter_sweep.py --input path/to/mono.wav
    python tests/run_parameter_sweep.py --input path/to/mono.wav --configs test_baseline test_tolerance30Hz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sms import DEFAULT_SMS_PARAMS, SmsParameters, export_wav, load_mono_audio, spectral_modeling_synthesis

TEST_DIR = Path(__file__).resolve().parent
F_CHAOS_HZ = 2000.0


class SweepConfig:
    def __init__(self, folder_name: str, description: str, overrides: dict[str, Any]):
        self.folder_name = folder_name
        self.description = description
        self.overrides = overrides


LOGICAL_CONFIGS: list[SweepConfig] = [
    SweepConfig("test_baseline", "Paper defaults from config/sms_defaults.yaml", {}),
    SweepConfig(
        "test_tolerance30Hz",
        "Stricter peak continuation: max_partial_frequency_deviation_hz=30",
        {"max_partial_frequency_deviation_hz": 30.0},
    ),
    SweepConfig(
        "test_missedFrames1",
        "Tighter guide survival: max_consecutive_missed_frames=1",
        {"max_consecutive_missed_frames": 1},
    ),
    SweepConfig(
        "test_threshold40dB",
        "Fewer peaks per frame: peak_detection_threshold_db=-40",
        {"peak_detection_threshold_db": -40.0},
    ),
    SweepConfig(
        "test_envelope500Hz",
        "Wider envelope sections: envelope_section_bandwidth_hz=500",
        {"envelope_section_bandwidth_hz": 500.0},
    ),
]


def _high_band_partial_count(analysis, f_min_hz: float = F_CHAOS_HZ) -> int:
    return sum(
        1 for p in analysis.partials if float(np.max(p.frequencies_hz)) >= f_min_hz
    )


def _apply_overrides(params: SmsParameters, overrides: dict[str, Any]) -> SmsParameters:
    if not overrides:
        return params
    return replace(params, **overrides)


def _reconstruction_error(original: np.ndarray, resynth: np.ndarray) -> float:
    n = min(len(original), len(resynth))
    if n == 0:
        return float("nan")
    err = original[:n] - resynth[:n]
    return float(np.sqrt(np.mean(err**2)))


def run_sweep(input_wav: str, config_names: list[str] | None) -> None:
    original, sr = load_mono_audio(input_wav)
    selected = LOGICAL_CONFIGS
    if config_names:
        name_set = set(config_names)
        selected = [c for c in LOGICAL_CONFIGS if c.folder_name in name_set]
        missing = name_set - {c.folder_name for c in selected}
        if missing:
            raise ValueError(f"Unknown config(s): {sorted(missing)}")

    sweep_root = TEST_DIR / "sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    print(f"Parameter sweep on {input_wav}")
    for cfg in selected:
        params = _apply_overrides(DEFAULT_SMS_PARAMS, cfg.overrides)
        out_dir = sweep_root / cfg.folder_name
        audio_dir = out_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        det, stoch, full, analysis = spectral_modeling_synthesis(original, sr, params=params)
        elapsed = time.perf_counter() - t0

        export_wav(str(audio_dir / "deterministic.wav"), det, sr)
        export_wav(str(audio_dir / "stochastic.wav"), stoch, sr)
        export_wav(str(audio_dir / "resynthesis.wav"), full, sr)

        metrics = {
            "config": cfg.folder_name,
            "description": cfg.description,
            "overrides": cfg.overrides,
            "f0_hz": analysis.f0_hz,
            "partial_count": len(analysis.partials),
            "high_band_partial_count": _high_band_partial_count(analysis),
            "rmse": _reconstruction_error(original, full),
            "elapsed_s": elapsed,
            "parameters": asdict(params),
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary.append(metrics)
        print(
            f"  {cfg.folder_name}: k={metrics['partial_count']}, "
            f"RMSE={metrics['rmse']:.6f}, {elapsed:.2f}s"
        )

    (sweep_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote sweep results under {sweep_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SMS parameter sweep.")
    parser.add_argument("--input", required=True, help="Mono WAV path")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Config folder names to run (default: all)",
    )
    args = parser.parse_args()
    run_sweep(args.input, args.configs)


if __name__ == "__main__":
    main()
