"""Mono WAV file I/O."""

from __future__ import annotations

import os

import numpy as np
from scipy.io import wavfile


def load_mono_audio(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file and return a float64 mono signal in [-1, 1]."""
    sample_rate, data = wavfile.read(path)

    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        peak = np.iinfo(data.dtype).max
        data = data.astype(np.float64) / peak
    else:
        data = data.astype(np.float64)

    return data, int(sample_rate)


def export_wav(path: str, signal_x: np.ndarray, sample_rate_hz: int) -> None:
    """Write a float32 mono WAV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wavfile.write(path, sample_rate_hz, signal_x.astype(np.float32))
