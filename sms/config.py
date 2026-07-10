"""Load SMS parameters from YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _PACKAGE_ROOT / "config" / "sms_defaults.yaml"


@dataclass(frozen=True)
class SmsParameters:
    """Named configuration for analysis and resynthesis (Serra & Smith, 1990)."""

    sample_rate_hz: float = 44100.0
    n_fft: int = 2048
    hop_size: int = 512
    analysis_window: str = "blackman_harris"
    analysis_window_beta: float | None = None

    peak_detection_threshold_db: float = -50.0
    min_peak_separation_bins: int = 4
    max_peaks_per_frame: int = 40

    max_partial_frequency_deviation_hz: float = 80.0
    max_consecutive_missed_frames: int = 3
    harmonic_association_tolerance: float = 0.07
    min_partial_duration_frames: int = 3

    f0_search_lo_hz: float = 80.0
    f0_search_hi_hz: float = 200.0

    envelope_section_bandwidth_hz: float = 350.0
    harmonic_notch_radius_bins: int = 2

    stochastic_grain_length: int = 2048
    overlap_add_hop: int = 512
    random_seed: int = 42

    @property
    def frequency_resolution_hz(self) -> float:
        return self.sample_rate_hz / self.n_fft

    @property
    def frame_period_s(self) -> float:
        return self.hop_size / self.sample_rate_hz

    @property
    def analysis_window_display_name(self) -> str:
        key = self.analysis_window.strip().lower().replace(" ", "_")
        if key == "kaiser" and self.analysis_window_beta is not None:
            return f"Kaiser (beta={self.analysis_window_beta:g})"
        return key.replace("_", " ").title()


def _get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_config(path: str | Path | None = None) -> SmsParameters:
    """Load ``SmsParameters`` from a YAML file (defaults to ``config/sms_defaults.yaml``)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    kaiser_beta = _get_nested(data, "analysis", "kaiser_beta")
    return SmsParameters(
        sample_rate_hz=float(data.get("sample_rate_hz", 44100.0)),
        n_fft=int(_get_nested(data, "analysis", "n_fft", default=2048)),
        hop_size=int(_get_nested(data, "analysis", "hop_size", default=512)),
        analysis_window=str(_get_nested(data, "analysis", "window", default="blackman_harris")),
        analysis_window_beta=float(kaiser_beta) if kaiser_beta is not None else None,
        peak_detection_threshold_db=float(
            _get_nested(data, "peak_detection", "threshold_db", default=-50.0)
        ),
        min_peak_separation_bins=int(
            _get_nested(data, "peak_detection", "min_separation_bins", default=4)
        ),
        max_peaks_per_frame=int(
            _get_nested(data, "peak_detection", "max_peaks_per_frame", default=40)
        ),
        max_partial_frequency_deviation_hz=float(
            _get_nested(data, "continuation", "max_frequency_deviation_hz", default=80.0)
        ),
        max_consecutive_missed_frames=int(
            _get_nested(data, "continuation", "max_missed_frames", default=3)
        ),
        harmonic_association_tolerance=float(
            _get_nested(data, "continuation", "harmonic_tolerance", default=0.07)
        ),
        min_partial_duration_frames=int(
            _get_nested(data, "continuation", "min_duration_frames", default=3)
        ),
        f0_search_lo_hz=float(_get_nested(data, "f0_estimation", "search_lo_hz", default=80.0)),
        f0_search_hi_hz=float(_get_nested(data, "f0_estimation", "search_hi_hz", default=200.0)),
        envelope_section_bandwidth_hz=float(
            _get_nested(data, "stochastic", "envelope_section_bandwidth_hz", default=350.0)
        ),
        harmonic_notch_radius_bins=int(
            _get_nested(data, "stochastic", "harmonic_notch_radius_bins", default=2)
        ),
        stochastic_grain_length=int(_get_nested(data, "stochastic", "grain_length", default=2048)),
        overlap_add_hop=int(_get_nested(data, "stochastic", "overlap_add_hop", default=512)),
        random_seed=int(data.get("random_seed", 42)),
    )


DEFAULT_SMS_PARAMS = load_config()
