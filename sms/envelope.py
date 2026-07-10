"""
Stochastic envelope estimation for SMS (Serra & Smith, 1990, Fig. 7).

Magnitude subtraction, harmonic notch masking, and piecewise-linear envelope.
"""

from __future__ import annotations

import numpy as np

F_MIN_HZ = 0.0
F_MAX_HZ = 8000.0
DB_MIN = -80.0
LIN_FLOOR = 10.0 ** (DB_MIN / 20.0)


def lin_to_db(magnitude_lin: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(magnitude_lin, 1e-12))


def build_notch_mask(
    n_bins: int,
    peak_indices: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    """Mark bins excluded from envelope control-point fitting."""
    notch = np.zeros(n_bins, dtype=bool)
    for peak in peak_indices:
        lo = max(0, int(peak) - radius)
        hi = min(n_bins, int(peak) + radius + 1)
        notch[lo:hi] = True
    return notch


def compute_residual(mag_lin: np.ndarray, d_lin: np.ndarray) -> np.ndarray:
    """|E_l(k)| = ||X_l(k)| - |D_l(k)|| (magnitude subtraction)."""
    return np.abs(mag_lin - d_lin)


def piecewise_linear_envelope(
    freqs_hz: np.ndarray,
    residual_lin: np.ndarray,
    notch_mask: np.ndarray,
    sample_rate: int,
    section_hz: float = 350.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SMS envelope: local maximum per fixed frequency section, linear segments.

    Bins marked in ``notch_mask`` are excluded so control points ride on the
    stochastic noise floor, not in harmonic notches.
    """
    valid = (~notch_mask) & (residual_lin > LIN_FLOOR * 1.5)

    n_sections = max(1, int(np.ceil((F_MAX_HZ - F_MIN_HZ) / section_hz)))
    edges = np.linspace(F_MIN_HZ, F_MAX_HZ, n_sections + 1)

    ctrl_freqs: list[float] = []
    ctrl_db: list[float] = []

    for sec in range(n_sections):
        f_lo, f_hi = edges[sec], edges[sec + 1]
        band = (freqs_hz >= f_lo) & (freqs_hz < f_hi) & valid
        if sec == n_sections - 1:
            band = (freqs_hz >= f_lo) & (freqs_hz <= f_hi) & valid

        if not np.any(band):
            continue

        section_lin = residual_lin[band]
        section_f = freqs_hz[band]
        peak_idx = int(np.argmax(section_lin))
        ctrl_freqs.append(float(section_f[peak_idx]))
        ctrl_db.append(float(lin_to_db(np.array([section_lin[peak_idx]]))[0]))

    if len(ctrl_freqs) == 0:
        raise RuntimeError("Insufficient envelope control points after notch exclusion.")

    if ctrl_freqs[0] > F_MIN_HZ:
        ctrl_freqs.insert(0, F_MIN_HZ)
        ctrl_db.insert(0, ctrl_db[0])
    if ctrl_freqs[-1] < F_MAX_HZ:
        ctrl_freqs.append(F_MAX_HZ)
        ctrl_db.append(ctrl_db[-1])

    ctrl_freqs.append(sample_rate / 2.0)
    ctrl_db.append(DB_MIN)

    ctrl_f = np.asarray(ctrl_freqs)
    ctrl_db_arr = np.asarray(ctrl_db)
    envelope_db = np.interp(freqs_hz, ctrl_f, ctrl_db_arr)
    return ctrl_f, ctrl_db_arr, envelope_db
