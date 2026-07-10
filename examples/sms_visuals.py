"""
Shared visualization helpers for SMS Studio and CLI plotting scripts.

Used by:
    examples/sms_gui.py
    examples/plot_decomposition.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from sms.analysis import SmsAnalysisResult

WAVEFORM_POINTS = 180


def waveform_envelope(audio: np.ndarray | None, points: int = WAVEFORM_POINTS) -> list[float]:
    """Downsample audio to a normalized peak envelope for lightweight canvas drawing."""
    if audio is None or len(audio) == 0:
        return []
    step = max(1, len(audio) // points)
    chunk = audio[::step]
    peak = float(np.max(np.abs(chunk))) or 1.0
    return (chunk / peak).tolist()


def narrowband_spectrogram(
    x: np.ndarray,
    sample_rate_hz: int,
    n_fft: int = 2048,
    hop_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a narrowband magnitude spectrogram in dB."""
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


def create_decomposition_figure(
    original: np.ndarray,
    deterministic: np.ndarray,
    stochastic: np.ndarray,
    sample_rate_hz: int,
    analysis: SmsAnalysisResult,
    *,
    fmax_hz: float = 8000.0,
    n_fft: int = 2048,
    hop_size: int = 512,
) -> Figure:
    """Build the three-panel decomposition spectrogram figure."""
    import matplotlib.pyplot as plt

    branches = [
        ("Original", original),
        ("Deterministic", deterministic),
        ("Stochastic", stochastic),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    mesh = None
    for ax, (title, sig) in zip(axes, branches, strict=True):
        freqs_hz, times_s, spec_db = narrowband_spectrogram(
            sig, sample_rate_hz, n_fft=n_fft, hop_size=hop_size
        )
        mask = freqs_hz <= fmax_hz
        mesh = ax.pcolormesh(
            times_s,
            freqs_hz[mask] / 1000.0,
            spec_db[mask, :],
            shading="gouraud",
            cmap="magma",
            vmin=-80,
            vmax=0,
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")

    fig.colorbar(mesh, ax=axes, label="Magnitude (dB)", fraction=0.02, pad=0.02)
    fig.suptitle(
        f"SMS decomposition — f₀ ≈ {analysis.f0_hz:.1f} Hz, k = {len(analysis.partials)} partials",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def save_decomposition_figure(
    original: np.ndarray,
    deterministic: np.ndarray,
    stochastic: np.ndarray,
    sample_rate_hz: int,
    analysis: SmsAnalysisResult,
    output_path: str | Path,
    **kwargs,
) -> Path:
    """Render and save the decomposition spectrogram figure."""
    import matplotlib.pyplot as plt

    fig = create_decomposition_figure(
        original,
        deterministic,
        stochastic,
        sample_rate_hz,
        analysis,
        **kwargs,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
