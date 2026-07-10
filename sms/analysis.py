"""
Deterministic analysis for Spectral Modeling Synthesis (Serra & Smith, 1990).

STFT magnitude, log-domain peak detection, parabolic interpolation, and greedy
peak continuation (frequency guides).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from sms.config import DEFAULT_SMS_PARAMS, SmsParameters

_WINDOW_ALIASES: dict[str, str] = {
    "blackman_harris": "blackmanharris",
    "blackman-harris": "blackmanharris",
    "blackmanharris": "blackmanharris",
    "hann": "hann",
    "hanning": "hann",
    "hamming": "hamming",
    "blackman": "blackman",
    "kaiser": "kaiser",
}


@dataclass
class RefinedPeak:
    """Parabolically refined spectral peak in one STFT frame."""

    bin_index: float
    frequency_hz: float
    amplitude: float


@dataclass
class FrequencyGuide:
    """Tracks a single partial trajectory across STFT frames."""

    times_s: list[float] = field(default_factory=list)
    freqs_hz: list[float] = field(default_factory=list)
    amps: list[float] = field(default_factory=list)
    last_freq_hz: float = 0.0
    missed_frames: int = 0
    active: bool = True


@dataclass
class SinusoidalPartial:
    """One tracked partial r: times t, frequency f_r(t), amplitude A_r(t)."""

    times_s: np.ndarray
    frequencies_hz: np.ndarray
    amplitudes: np.ndarray


@dataclass
class SmsAnalysisResult:
    partials: list[SinusoidalPartial]
    f0_hz: float
    frame_times_s: np.ndarray
    frequency_bins_hz: np.ndarray
    stft_magnitude: np.ndarray
    frequency_guides: list[FrequencyGuide]
    sample_rate_hz: int
    parameters: SmsParameters


def stft_magnitude_scale(n_fft: int) -> float:
    """One-sided rFFT magnitude normalization used throughout the pipeline."""
    return n_fft / 2.0


def make_analysis_window(
    n_fft: int,
    window_name: str = "blackman_harris",
    *,
    beta: float | None = None,
) -> np.ndarray:
    """Build a symmetric analysis window for STFT peak tracking."""
    key = window_name.strip().lower().replace(" ", "_")
    kind = _WINDOW_ALIASES.get(key)
    if kind is None:
        supported = ", ".join(sorted({k for k in _WINDOW_ALIASES if "_" not in k or k == "blackman_harris"}))
        raise ValueError(f"Unknown analysis window {window_name!r}. Supported: {supported}, kaiser")

    if kind == "blackmanharris":
        return signal.windows.blackmanharris(n_fft, sym=True)
    if kind == "hann":
        return signal.windows.hann(n_fft, sym=True)
    if kind == "hamming":
        return signal.windows.hamming(n_fft, sym=True)
    if kind == "blackman":
        return signal.windows.blackman(n_fft, sym=True)
    if kind == "kaiser":
        if beta is None:
            raise ValueError("Kaiser window requires beta= (e.g. beta=8.0)")
        return signal.windows.kaiser(n_fft, beta=beta, sym=True)
    raise ValueError(f"Unhandled window kind {kind!r}")


def analysis_window_for_params(params: SmsParameters) -> np.ndarray:
    """Return the analysis window array implied by ``params``."""
    return make_analysis_window(
        params.n_fft,
        params.analysis_window,
        beta=params.analysis_window_beta,
    )


def compute_stft_magnitude(
    x: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_size: int,
    window: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return magnitude STFT, frame center times (s), and rFFT frequency axis (Hz).

    Frames are zero-phase centered: sample at frame center ``l * hop`` sits at
    the window midpoint (with edge zero-padding).
    """
    if window is None:
        window = signal.windows.blackmanharris(n_fft, sym=True)
    elif len(window) != n_fft:
        raise ValueError(f"window length {len(window)} != n_fft {n_fft}")

    scale = stft_magnitude_scale(n_fft)
    n_frames = 1 + max(0, (len(x) - 1) // hop_size)
    n_bins = n_fft // 2 + 1
    magnitude = np.zeros((n_frames, n_bins), dtype=np.float64)
    half = n_fft // 2

    for frame_idx in range(n_frames):
        center = frame_idx * hop_size
        start = center - half
        frame = np.zeros(n_fft, dtype=np.float64)
        src_lo = max(0, start)
        src_hi = min(len(x), start + n_fft)
        dst_lo = src_lo - start
        frame[dst_lo : dst_lo + (src_hi - src_lo)] = x[src_lo:src_hi]
        frame *= window
        spectrum = np.fft.rfft(frame, n=n_fft)
        magnitude[frame_idx, :] = np.abs(spectrum) / scale

    freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    times_s = np.arange(n_frames, dtype=np.float64) * hop_size / sample_rate
    return magnitude, times_s, freqs_hz


def parabolic_interpolation(
    magnitudes: np.ndarray,
    peak_bin: int,
) -> RefinedPeak | None:
    """Refine a discrete local maximum using a three-point parabolic fit."""
    if peak_bin < 1 or peak_bin >= len(magnitudes) - 1:
        return None

    alpha = magnitudes[peak_bin - 1]
    beta = magnitudes[peak_bin]
    gamma = magnitudes[peak_bin + 1]

    denominator = alpha - 2.0 * beta + gamma
    if abs(denominator) < 1e-12:
        return None

    p = 0.5 * (alpha - gamma) / denominator
    p = float(np.clip(p, -0.5, 0.5))

    refined_bin = peak_bin + p
    refined_amp = beta - 0.25 * (alpha - gamma) * p

    if refined_amp <= 0.0:
        return None

    return RefinedPeak(
        bin_index=refined_bin,
        frequency_hz=0.0,
        amplitude=refined_amp,
    )


def detect_peaks_in_frame(
    magnitudes: np.ndarray,
    freqs_hz: np.ndarray,
    threshold_db: float,
    min_distance_bins: int,
    max_peaks: int,
) -> list[RefinedPeak]:
    """Extract parabolically refined peaks from one STFT magnitude frame."""
    log_mag = 20.0 * np.log10(np.maximum(magnitudes, 1e-12))
    frame_peak_db = float(np.max(log_mag))
    height_threshold = frame_peak_db + threshold_db

    candidate_bins, _ = signal.find_peaks(
        log_mag,
        height=height_threshold,
        distance=min_distance_bins,
    )

    refined: list[RefinedPeak] = []
    for peak_bin in candidate_bins:
        result = parabolic_interpolation(magnitudes, int(peak_bin))
        if result is None:
            continue
        result.frequency_hz = result.bin_index * freqs_hz[1]
        refined.append(result)

    refined.sort(key=lambda p: p.amplitude, reverse=True)
    return refined[:max_peaks]


def _estimate_frame_f0_hz(
    active_guides: list[FrequencyGuide],
    default_f0_hz: float,
) -> float:
    estimates: list[float] = []
    for guide in active_guides:
        if len(guide.freqs_hz) < 1:
            continue
        median_f = float(np.median(guide.freqs_hz[-min(5, len(guide.freqs_hz)) :]))
        for h in range(1, 24):
            candidate_f0 = median_f / h
            if 75.0 <= candidate_f0 <= 240.0:
                estimates.append(candidate_f0)
                break
    if not estimates:
        return default_f0_hz
    return float(np.median(estimates))


def continue_peaks_greedy(
    peaks_per_frame: list[list[RefinedPeak]],
    times_s: np.ndarray,
    max_freq_deviation_hz: float,
    max_missed_frames: int,
    f0_hz: float | None = None,
    harmonic_tolerance_ratio: float = 0.07,
) -> list[FrequencyGuide]:
    """Link frame-wise peaks into partial trajectories using frequency guides."""
    guides: list[FrequencyGuide] = []
    default_f0 = f0_hz if f0_hz is not None else 130.0

    for frame_idx, frame_peaks in enumerate(peaks_per_frame):
        time_s = float(times_s[frame_idx])
        unmatched = list(frame_peaks)

        active_guides = sorted(
            [g for g in guides if g.active],
            key=lambda g: g.last_freq_hz,
        )
        frame_f0 = _estimate_frame_f0_hz(active_guides, default_f0) if f0_hz else default_f0

        for guide in active_guides:
            if not unmatched:
                guide.missed_frames += 1
                if guide.missed_frames > max_missed_frames:
                    guide.active = False
                continue

            if len(guide.freqs_hz) >= 2:
                predicted_hz = 2.0 * guide.last_freq_hz - guide.freqs_hz[-2]
            else:
                predicted_hz = guide.last_freq_hz

            if f0_hz is not None and guide.last_freq_hz > 800.0:
                harmonic = max(1, int(round(guide.last_freq_hz / frame_f0)))
                expected_hz = harmonic * frame_f0
                tol_hz = harmonic_tolerance_ratio * expected_hz + 0.5 * max_freq_deviation_hz
                harmonic_candidates = [
                    (idx, p)
                    for idx, p in enumerate(unmatched)
                    if abs(p.frequency_hz - expected_hz) <= tol_hz
                ]
                if harmonic_candidates:
                    best_idx, peak = min(
                        harmonic_candidates,
                        key=lambda item: abs(item[1].frequency_hz - predicted_hz),
                    )
                    unmatched.pop(best_idx)
                    guide.times_s.append(time_s)
                    guide.freqs_hz.append(peak.frequency_hz)
                    guide.amps.append(peak.amplitude)
                    guide.last_freq_hz = peak.frequency_hz
                    guide.missed_frames = 0
                    continue

            distances = [abs(p.frequency_hz - predicted_hz) for p in unmatched]
            best_idx = int(np.argmin(distances))

            if distances[best_idx] <= max_freq_deviation_hz:
                peak = unmatched.pop(best_idx)
                guide.times_s.append(time_s)
                guide.freqs_hz.append(peak.frequency_hz)
                guide.amps.append(peak.amplitude)
                guide.last_freq_hz = peak.frequency_hz
                guide.missed_frames = 0
            else:
                guide.missed_frames += 1
                if guide.missed_frames > max_missed_frames:
                    guide.active = False

        for peak in unmatched:
            guides.append(
                FrequencyGuide(
                    times_s=[time_s],
                    freqs_hz=[peak.frequency_hz],
                    amps=[peak.amplitude],
                    last_freq_hz=peak.frequency_hz,
                )
            )

    return [g for g in guides if len(g.times_s) >= 3]


def _estimate_f0_from_average_spectrum(
    stft_magnitude: np.ndarray,
    frequency_bins_hz: np.ndarray,
    params: SmsParameters,
) -> float:
    n_frames = stft_magnitude.shape[0]
    avg_spectrum = np.mean(stft_magnitude[max(1, n_frames // 8) :], axis=0)
    best_f0, best_score = 130.81, -1.0
    for candidate in np.linspace(params.f0_search_lo_hz, params.f0_search_hi_hz, 241):
        score = 0.0
        for harmonic in range(1, 28):
            target = harmonic * candidate
            if target > frequency_bins_hz[-1]:
                break
            idx = int(np.argmin(np.abs(frequency_bins_hz - target)))
            score += float(avg_spectrum[idx]) / np.sqrt(harmonic)
        if score > best_score:
            best_score, best_f0 = score, float(candidate)
    return best_f0


def _guides_to_partials(
    guides: list[FrequencyGuide],
    min_duration_frames: int,
) -> list[SinusoidalPartial]:
    return [
        SinusoidalPartial(
            times_s=np.asarray(g.times_s, dtype=np.float64),
            frequencies_hz=np.asarray(g.freqs_hz, dtype=np.float64),
            amplitudes=np.asarray(g.amps, dtype=np.float64),
        )
        for g in guides
        if len(g.times_s) >= min_duration_frames
    ]


def extract_sinusoidal_partials(
    signal_x: np.ndarray,
    sample_rate_hz: int,
    params: SmsParameters = DEFAULT_SMS_PARAMS,
) -> SmsAnalysisResult:
    """Deterministic analysis: |X_l(k)| peaks, parabolic refinement, peak continuation."""
    window = analysis_window_for_params(params)
    stft_mag, frame_times_s, freq_bins_hz = compute_stft_magnitude(
        signal_x,
        sample_rate_hz,
        params.n_fft,
        params.hop_size,
        window=window,
    )
    f0_hz = _estimate_f0_from_average_spectrum(stft_mag, freq_bins_hz, params)

    peaks_per_frame: list[list[RefinedPeak]] = []
    for frame_idx in range(stft_mag.shape[0]):
        peaks_per_frame.append(
            detect_peaks_in_frame(
                stft_mag[frame_idx],
                freq_bins_hz,
                threshold_db=params.peak_detection_threshold_db,
                min_distance_bins=params.min_peak_separation_bins,
                max_peaks=params.max_peaks_per_frame,
            )
        )

    guides = continue_peaks_greedy(
        peaks_per_frame,
        frame_times_s,
        max_freq_deviation_hz=params.max_partial_frequency_deviation_hz,
        max_missed_frames=params.max_consecutive_missed_frames,
        f0_hz=f0_hz,
        harmonic_tolerance_ratio=params.harmonic_association_tolerance,
    )

    return SmsAnalysisResult(
        partials=_guides_to_partials(guides, params.min_partial_duration_frames),
        f0_hz=f0_hz,
        frame_times_s=frame_times_s,
        frequency_bins_hz=freq_bins_hz,
        stft_magnitude=stft_mag,
        frequency_guides=guides,
        sample_rate_hz=sample_rate_hz,
        parameters=params,
    )
