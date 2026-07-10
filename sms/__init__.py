"""
Spectral Modeling Synthesis — Serra & Smith (1990).

Python implementation of the deterministic plus stochastic decomposition:
  s(t) = d(t) + e(t)
"""

from sms.analysis import (
    FrequencyGuide,
    RefinedPeak,
    SinusoidalPartial,
    SmsAnalysisResult,
    compute_stft_magnitude,
    continue_peaks_greedy,
    detect_peaks_in_frame,
    extract_sinusoidal_partials,
    parabolic_interpolation,
)
from sms.audio_io import export_wav, load_mono_audio
from sms.config import DEFAULT_SMS_PARAMS, SmsParameters, load_config
from sms.synthesis import (
    compute_stochastic_envelope_frame,
    spectral_modeling_synthesis,
    synthesize_sinusoidal_component,
    synthesize_stochastic_component,
)

__all__ = [
    "DEFAULT_SMS_PARAMS",
    "FrequencyGuide",
    "RefinedPeak",
    "SinusoidalPartial",
    "SmsAnalysisResult",
    "SmsParameters",
    "compute_stft_magnitude",
    "compute_stochastic_envelope_frame",
    "continue_peaks_greedy",
    "detect_peaks_in_frame",
    "export_wav",
    "extract_sinusoidal_partials",
    "load_config",
    "load_mono_audio",
    "parabolic_interpolation",
    "spectral_modeling_synthesis",
    "synthesize_sinusoidal_component",
    "synthesize_stochastic_component",
]
