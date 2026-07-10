"""Mono WAV file I/O."""

from __future__ import annotations

import os

import numpy as np
from scipy.io import wavfile


def load_mono_audio(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file and return a float64 mono signal in [-1, 1]."""
    try:
        sample_rate, data = wavfile.read(path)
    except Exception as exc:
        raise ValueError(f"Could not read WAV file: {exc}") from exc

    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float64)
        peak = np.max(np.abs(data))
        if peak > 1.0:
            data = data / peak
        return data, int(sample_rate)

    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"Unsupported WAV sample type: {data.dtype}")

    info = np.iinfo(data.dtype)
    data = data.astype(np.float64) / max(abs(info.min), info.max)
    return data, int(sample_rate)


def export_wav(path: str, signal_x: np.ndarray, sample_rate_hz: int) -> None:
    """Write a float32 mono WAV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    clipped = np.clip(signal_x.astype(np.float32), -1.0, 1.0)
    wavfile.write(path, sample_rate_hz, clipped)
