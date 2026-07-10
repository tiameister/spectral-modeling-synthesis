"""
Full Spectral Modeling Synthesis pipeline (Serra & Smith, 1990).

Deterministic additive sinusoids plus stochastic filtered-noise residual.
"""

from __future__ import annotations

import numpy as np

from sms.analysis import (
    SinusoidalPartial,
    SmsAnalysisResult,
    analysis_window_for_params,
    compute_stft_magnitude,
    extract_sinusoidal_partials,
    stft_magnitude_scale,
)
from sms.config import DEFAULT_SMS_PARAMS, SmsParameters
from sms.envelope import build_notch_mask, piecewise_linear_envelope

_ONE_SIDED_PARSEVAL_GAIN = np.sqrt(2.0)


def _peak_to_wave_scale(params: SmsParameters) -> float:
    window = analysis_window_for_params(params)
    return float(params.n_fft) / float(np.sum(window))


def _stochastic_window(params: SmsParameters) -> np.ndarray:
    return np.hanning(params.stochastic_grain_length)


def _synthesize_one_partial(
    n_samples: int,
    sample_rate_hz: int,
    times_s: np.ndarray,
    frequencies_hz: np.ndarray,
    amplitudes: np.ndarray,
    peak_to_wave: float,
) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    a_t = np.interp(t, times_s, amplitudes * peak_to_wave, left=0.0, right=0.0)
    f_t = np.interp(t, times_s, frequencies_hz, left=frequencies_hz[0], right=frequencies_hz[-1])
    omega = 2.0 * np.pi * f_t
    phase = np.zeros(n_samples, dtype=np.float64)
    for i in range(1, n_samples):
        phase[i] = phase[i - 1] + 0.5 * (omega[i] + omega[i - 1]) / sample_rate_hz
    return a_t * np.cos(phase)


def synthesize_sinusoidal_component(
    n_samples: int,
    sample_rate_hz: int,
    partials: list[SinusoidalPartial],
    params: SmsParameters = DEFAULT_SMS_PARAMS,
) -> np.ndarray:
    """d(t) = sum_r A_r(t) cos[theta_r(t)] with linear A, f and trapezoidal phase."""
    peak_to_wave = _peak_to_wave_scale(params)
    out = np.zeros(n_samples, dtype=np.float64)
    for partial in partials:
        out += _synthesize_one_partial(
            n_samples,
            sample_rate_hz,
            partial.times_s,
            partial.frequencies_hz,
            partial.amplitudes,
            peak_to_wave,
        )
    return out


def _partial_bin_indices_at_frame(
    partials: list[SinusoidalPartial],
    frame_time_s: float,
    frequency_bins_hz: np.ndarray,
) -> np.ndarray:
    indices: list[int] = []
    for partial in partials:
        f_hz = float(
            np.interp(
                frame_time_s,
                partial.times_s,
                partial.frequencies_hz,
                left=np.nan,
                right=np.nan,
            )
        )
        if np.isfinite(f_hz):
            indices.append(int(np.argmin(np.abs(frequency_bins_hz - f_hz))))
    if not indices:
        return np.array([], dtype=np.int64)
    return np.unique(np.asarray(indices, dtype=np.int64))


def compute_stochastic_envelope_frame(
    x_l: np.ndarray,
    d_l: np.ndarray,
    frequency_bins_hz: np.ndarray,
    sample_rate_hz: int,
    partials: list[SinusoidalPartial],
    frame_time_s: float,
    params: SmsParameters = DEFAULT_SMS_PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """|E_l| = ||X_l| - |D_l|| and piecewise-linear envelope (Fig. 7)."""
    e_l = np.abs(x_l - d_l)
    peak_indices = _partial_bin_indices_at_frame(partials, frame_time_s, frequency_bins_hz)
    notch = build_notch_mask(len(x_l), peak_indices, radius=params.harmonic_notch_radius_bins)
    try:
        _, _, envelope_db = piecewise_linear_envelope(
            frequency_bins_hz,
            e_l,
            notch,
            int(sample_rate_hz),
            section_hz=params.envelope_section_bandwidth_hz,
        )
    except RuntimeError:
        envelope_db = 20.0 * np.log10(np.maximum(e_l, 1e-12))
    envelope_lin = 10.0 ** (envelope_db / 20.0)
    return e_l, d_l, envelope_lin


def _envelope_rms(envelope_lin: np.ndarray) -> float:
    return float(np.sqrt(np.sum(envelope_lin**2) / 2.0))


def _synthesize_noise_grain(
    envelope_lin: np.ndarray,
    rng: np.random.Generator,
    params: SmsParameters,
) -> np.ndarray:
    mag_scale = stft_magnitude_scale(params.n_fft)
    stoch_window = _stochastic_window(params)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=len(envelope_lin))
    spectrum = envelope_lin * _ONE_SIDED_PARSEVAL_GAIN * mag_scale * np.exp(1j * phase)
    grain = np.fft.irfft(spectrum, n=params.n_fft).real[: params.stochastic_grain_length]
    target_rms = _envelope_rms(envelope_lin)
    grain_rms = float(np.sqrt(np.mean(grain**2)))
    if grain_rms > 1e-12:
        grain *= target_rms / grain_rms
    return grain * stoch_window


def _ola_normalizer(
    n_samples: int,
    grain_len: int,
    hop: int,
    n_frames: int,
    params: SmsParameters,
) -> np.ndarray:
    window = _stochastic_window(params) if grain_len == params.stochastic_grain_length else np.hanning(grain_len)
    pad = grain_len // 2
    norm = np.zeros(n_samples + 2 * pad, dtype=np.float64)
    for frame_idx in range(n_frames):
        center = pad + frame_idx * hop
        start = center - grain_len // 2
        norm[start : start + grain_len] += window**2
    norm = norm[pad : pad + n_samples]
    return np.sqrt(np.maximum(norm, 1e-12))


def synthesize_stochastic_component(
    original: np.ndarray,
    sinusoidal: np.ndarray,
    analysis: SmsAnalysisResult,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize e(t) from magnitude-subtracted residual envelopes."""
    params = analysis.parameters
    sr = analysis.sample_rate_hz
    window = analysis_window_for_params(params)
    x_mag, _, _ = compute_stft_magnitude(original, sr, params.n_fft, params.hop_size, window=window)
    d_mag, _, _ = compute_stft_magnitude(sinusoidal, sr, params.n_fft, params.hop_size, window=window)
    n_frames = min(x_mag.shape[0], d_mag.shape[0])

    grain_len = params.stochastic_grain_length
    hop = params.overlap_add_hop
    pad = grain_len // 2
    output = np.zeros(len(original) + 2 * pad, dtype=np.float64)

    for frame_idx in range(n_frames):
        t_s = float(analysis.frame_times_s[min(frame_idx, len(analysis.frame_times_s) - 1)])
        _, _, envelope_lin = compute_stochastic_envelope_frame(
            x_mag[frame_idx],
            d_mag[frame_idx],
            analysis.frequency_bins_hz,
            sr,
            analysis.partials,
            t_s,
            params,
        )
        grain = _synthesize_noise_grain(envelope_lin, rng, params)
        center = pad + frame_idx * hop
        start = center - grain_len // 2
        output[start : start + grain_len] += grain

    output = output[pad : pad + len(original)]
    norm = _ola_normalizer(len(original), grain_len, hop, n_frames, params)
    return output / np.maximum(norm, 1e-12)


def spectral_modeling_synthesis(
    signal_x: np.ndarray,
    sample_rate_hz: int,
    rng: np.random.Generator | None = None,
    params: SmsParameters = DEFAULT_SMS_PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SmsAnalysisResult]:
    """
    Full SMS: extract partials, synthesize d(t) and e(t), return (d, e, d+e, analysis).
    """
    if rng is None:
        rng = np.random.default_rng(params.random_seed)

    analysis = extract_sinusoidal_partials(signal_x, sample_rate_hz, params)
    d_t = synthesize_sinusoidal_component(len(signal_x), sample_rate_hz, analysis.partials, params)
    e_t = synthesize_stochastic_component(signal_x, d_t, analysis, rng)
    n = min(len(signal_x), len(d_t), len(e_t))
    s_t = d_t[:n] + e_t[:n]
    return d_t[:n], e_t[:n], s_t[:n], analysis
