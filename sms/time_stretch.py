"""
SMS time-stretch demo helpers (Serra & Smith musical-control motivation).

Preserves the initial stochastic attack while stretching the deterministic
component and stochastic tail independently.
"""

from __future__ import annotations

import numpy as np

from sms.config import DEFAULT_SMS_PARAMS


def resample_signal(x: np.ndarray, factor: float) -> np.ndarray:
    """Linear interpolation resampling by an arbitrary factor."""
    if factor == 1.0:
        return x.copy()
    n_out = max(2, int(np.round(len(x) * factor)))
    x_old = np.arange(len(x), dtype=np.float64)
    x_new = np.linspace(0, len(x) - 1, n_out)
    return np.interp(x_new, x_old, x)


def naive_phase_vocoder_stretch(
    original: np.ndarray,
    stretch_factor: float,
    sample_rate_hz: float = DEFAULT_SMS_PARAMS.sample_rate_hz,
) -> np.ndarray:
    """
    Simulate naive time-stretch smearing: resample entire waveform and
    diffuse attack energy with a wide Gaussian kernel.
    """
    stretched = resample_signal(original, stretch_factor)

    blur_len = min(int(0.012 * sample_rate_hz * stretch_factor), len(stretched) // 2)
    if blur_len > 4:
        sigma = blur_len * 0.22
        kernel_len = min(int(sigma * 8), blur_len)
        if kernel_len % 2 == 0:
            kernel_len -= 1
        kernel_len = max(kernel_len, 5)
        kx = np.arange(kernel_len) - kernel_len // 2
        kernel = np.exp(-0.5 * (kx / sigma) ** 2)
        kernel /= kernel.sum()
        smeared = np.convolve(stretched, kernel, mode="same")

        blend_len = min(int(0.06 * sample_rate_hz * stretch_factor), len(stretched))
        alpha = np.linspace(0.85, 0.0, blend_len) ** 0.6
        stretched[:blend_len] = alpha * smeared[:blend_len] + (1.0 - alpha) * stretched[:blend_len]

    return stretched


def sms_time_stretch(
    det: np.ndarray,
    stoch: np.ndarray,
    attack_len: int,
    stretch_factor: float,
) -> np.ndarray:
    """
    SMS time-stretch: stretch deterministic component and stochastic tail;
    preserve the initial stochastic attack at original duration.
    """
    stoch_attack = stoch[:attack_len]
    stoch_tail = stoch[attack_len:]

    det_stretched = resample_signal(det, stretch_factor)
    stoch_tail_stretched = resample_signal(stoch_tail, stretch_factor)

    total_len = max(len(det_stretched), attack_len + len(stoch_tail_stretched))
    result = np.zeros(total_len, dtype=np.float64)
    result[:attack_len] += stoch_attack
    result[attack_len : attack_len + len(stoch_tail_stretched)] += stoch_tail_stretched
    result[: len(det_stretched)] += det_stretched
    return result
