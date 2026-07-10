# Spectral Modeling Synthesis — Technical Study Report

**Project:** Python implementation of Serra & Smith (1990) Spectral Modeling Synthesis  
**Reference:** Serra & Smith, *Spectral Modeling Synthesis of Sounds*, *Computer Music Journal*, 1990  
**Implementation:** `sms/analysis.py`, `sms/synthesis.py`, `sms/envelope.py`  
**Configuration:** `config/sms_defaults.yaml`

This document is a study companion for the project. It collects the core equations, every important parameter, pipeline stages, and likely Q&A answers.

---

## 1. Core idea (one paragraph)

SMS decomposes a signal into **deterministic sinusoidal partials** (tones with tracked amplitude and frequency) plus a **stochastic residual** (filtered noise). The model assumes **linear superposition**:

\[
s(t) = \sum_{r=1}^{k} A_r(t)\cos\!\bigl[\theta_r(t)\bigr] + e(t) = d(t) + e(t)
\]

The deterministic branch captures horizontal ridges in a spectrogram (harmonic partials). The stochastic branch captures broadband energy the sinusoids cannot explain (pick attack, friction, breath). Both branches are resynthesized separately and summed in the time domain.

---

## 2. Slide-by-slide technical map

| Section | Slide topic | Key technical point |
|---------|-------------|---------------------|
| Intro | C3 guitar listen | Target: resynthesize `audio/clean_guitar_slide.wav` |
| Intro | Anatomy of sound | \(s =\) tones \(+\) noise; listen to det/stoch branches |
| Intro | Motivation | Sinusoids are inefficient for wideband transients |
| Intro | Spectrogram decomposition | Three-panel visual: original / det / stoch |
| Analysis | The model | ~200 partials on clean C3; residual = filtered white noise |
| Analysis | Deterministic pipeline | STFT peaks → parabolic refinement → peak continuation |
| Analysis | Parabolic interpolation | Sub-bin frequency from 3 log-magnitude bins |
| Analysis | Partial trajectories | Greedy frequency guides; 80 Hz tolerance, 3-frame sleep |
| Analysis | Magnitude subtraction | \(\|E_\ell(k)\| = \bigl\|\|X_\ell(k)\| - \|D_\ell(k)\|\bigr\|\) |
| Analysis | Visual subtraction | Attack burst (~180 ms) isolated in residual |
| Synthesis | Architecture | Dual path: oscillators + envelope-shaped noise OLA |
| Synthesis | Mechanics | Trapezoidal phase integration; Hanning grains, hop 512 |
| Synthesis | Full reconstruction | Not bit-perfect (phase discarded on residual) |
| Synthesis | Musical control | Independent time-stretch / pitch-shift / cross-synthesis |
| Synthesis | Time-stretch demo | 3× stretch; attack locked on stochastic branch |
| Limitations | Superposition collapse | Linear model fails under heavy distortion |
| Limitations | Gojira chord | ~900 partials; intermodulation breaks tracking |
| Conclusion | SMS vs DDSP | Same additive–noise synthesizer; DDSP learns parameters |

---

## 3. Analysis pipeline (deterministic branch)

### Stage 1 — STFT magnitude

- Input: mono audio \(x[n]\), sample rate \(f_s\).
- Window: **Blackman–Harris**, symmetric, length \(N_\mathrm{FFT} = 2048\).
- Hop: \(H = 512\) samples → **75% overlap**, frame period \(H/f_s \approx 11.6\,\mathrm{ms}\).
- Output per frame \(\ell\): one-sided magnitude spectrum \(|X_\ell(k)|\), \(k = 0 \ldots N/2\).
- Frequency resolution: \(\Delta f = f_s / N_\mathrm{FFT}\).

At \(f_s = 44100\,\mathrm{Hz}\): \(\Delta f \approx 21.53\,\mathrm{Hz}\) (shown on parabolic interpolation slide).

Frames are **zero-phase centered** (window midpoint aligned to sample index \(\ell H\)).

### Stage 2 — Peak detection

Per STFT frame:

1. Convert magnitudes to **log domain**: \(20\log_{10}|X_\ell(k)|\).
2. Find local maxima with `scipy.signal.find_peaks`.
3. Keep peaks above **frame peak + threshold** (default **−50 dB**).
4. Enforce **minimum separation** of **4 bins** between peaks.
5. Retain at most **40 peaks per frame** (strongest first).

Why log-magnitude? Matches Serra & Smith (1990): perceptually motivated, stable across dynamic range.

### Stage 3 — Parabolic interpolation

For discrete peak at bin \(k\), fit a parabola through magnitudes at bins \(k-1, k, k+1\):

\[
p = \frac{1}{2}\,\frac{\alpha - \gamma}{\alpha - 2\beta + \gamma}, \qquad
A_\mathrm{refined} = \beta - \frac{1}{4}(\alpha - \gamma)\,p
\]

where \(\alpha, \beta, \gamma\) are the three bin magnitudes and \(p\) is the sub-bin offset (clipped to \([-0.5, 0.5]\)).

Refined frequency: \(f = (k + p)\,\Delta f\).

**Why not just zero-pad?** Zero-padding increases FFT size and cost. Three-point parabolic fit gives sub-bin accuracy cheaply (exact under a Gaussian window assumption).

### Stage 4 — Peak continuation (greedy frequency guides)

Each partial is a **frequency guide** that links peaks across frames:

1. **Predict** next frequency: linear extrapolation from last two guide points, or last frequency if only one point.
2. **Match** closest unmatched peak within **±80 Hz** of prediction.
3. **Harmonic guard** (when \(f_0\) known): for guides above 800 Hz, prefer peaks near \(n \cdot f_0(t)\) with tolerance \(0.07 \cdot nf_0 + 0.5 \times 80\,\mathrm{Hz}\).
4. **Miss counter**: if no match for **3 consecutive frames**, guide is killed (“sleeping time”).
5. **New guides**: unmatched peaks seed new trajectories.
6. **Filter**: discard guides shorter than **3 frames**.

**Important:** This is **not** a fixed harmonic grid. Every strong spectral peak can become its own trajectory (~200 on clean C3, not ~22 harmonics).

### Stage 5 — f₀ estimation

Before continuation, estimate \(f_0\) from the **average magnitude spectrum** (excluding first 1/8 of frames):

- Search \(f_0 \in [80, 200]\,\mathrm{Hz}\) in 241 steps.
- Score each candidate by summing spectral energy at \(n f_0\) (harmonics 1–27), weighted by \(1/\sqrt{n}\).
- Default prior: 130.81 Hz (C3).

---

## 4. Stochastic branch

### Magnitude subtraction

Per analysis frame \(\ell\):

\[
|E_\ell(k)| = \bigl||X_\ell(k)| - |D_\ell(k)|\bigr|
\]

where \(|D_\ell(k)|\) is the STFT magnitude of the **synthesized deterministic signal** (not a theoretical model).

**Phase is discarded.** Residual is treated as **statistically diffuse**; perceptual content lives in the **spectral envelope**.

### Envelope estimation

1. Apply **harmonic notch mask** around tracked partial bin indices (radius **2 bins**) so envelope control points ignore tone energy.
2. Fit **piecewise-linear envelope** in dB with section bandwidth **350 Hz** (Serra & Smith Fig. 7 style).
3. Convert back to linear magnitude for synthesis.

### Stochastic synthesis (OLA)

Per frame:

1. Draw **random phases** uniform on \([0, 2\pi)\) for each frequency bin.
2. Build spectrum: envelope × random phase × normalization constants.
3. **Inverse RFFT** → time-domain grain.
4. **RMS-match** grain to envelope energy.
5. Apply **Hanning window**, length **2048** samples.
6. **Overlap-add** with hop **512** (same as analysis hop).
7. Divide by OLA normalizer \(\sqrt{\sum w^2}\) for stable output level.

Overlap factor: grain length / hop = 2048 / 512 = **4×** (paper convention \(M = 4H\)).

---

## 5. Deterministic synthesis

For each partial \(r\):

- **Amplitude** \(A_r(t)\): linear interpolation between frame values.
- **Frequency** \(f_r(t)\): linear interpolation between frame values.
- **Phase** \(\theta_r(t)\): **trapezoidal integration** of instantaneous frequency:
  \[
  \theta[n] = \theta[n-1] + \frac{\omega[n] + \omega[n-1]}{2 f_s}, \quad \omega = 2\pi f
  \]
- Output: \(d(t) = \sum_r A_r(t)\cos\theta_r(t)\).

Trapezoidal integration guarantees **phase continuity** → no clicks at frame boundaries.

Amplitude scaling uses window compensation factor `_PEAK_TO_WAVE` (Blackman–Harris sum normalization).

---

## 6. Complete parameter reference

All defaults live in `config/sms_defaults.yaml` and are loaded via `SmsParameters` in `sms/config.py`.

### 6.1 Time–frequency analysis

| Parameter | Symbol / field | Default | Meaning |
|-----------|----------------|---------|---------|
| Sample rate | `sample_rate_hz` | 44100 | Assumed for seminar assets |
| FFT size | `n_fft` | 2048 | Analysis & stochastic grain FFT size |
| Hop size | `hop_size` | 512 | Frame advance; 75% overlap |
| Analysis window | `analysis_window` | `blackman_harris` | ~92 dB sidelobe suppression |
| Frequency resolution | `f_s / N` | ~21.53 Hz @ 44.1 kHz | Bin spacing |
| Frame period | `H / f_s` | ~11.6 ms | Time between STFT frames |

### 6.2 Peak detection

| Parameter | Field | Default | Meaning |
|-----------|-------|---------|---------|
| Peak threshold | `peak_detection_threshold_db` | −50 dB | Relative to **frame** peak in log-mag |
| Min peak separation | `min_peak_separation_bins` | 4 | Suppress closely spaced spurious peaks |
| Max peaks / frame | `max_peaks_per_frame` | 40 | Cap before continuation |

### 6.3 Peak continuation

| Parameter | Field | Default | Meaning |
|-----------|-------|---------|---------|
| Max frequency deviation | `max_partial_frequency_deviation_hz` | 80 Hz | Guide–peak matching radius |
| Sleeping time | `max_consecutive_missed_frames` | 3 | Frames without match before guide dies |
| Harmonic tolerance | `harmonic_association_tolerance` | 0.07 | Relative tolerance for \(n f_0\) matching |
| Min partial length | `min_partial_duration_frames` | 3 | Drop short/unstable tracks |
| f₀ search range | `f0_search_lo/hi_hz` | 80 – 200 Hz | Initial f₀ estimation |
| C3 nominal | (presentation) | 130.81 Hz | Reference pitch for C3 guitar |

### 6.4 Stochastic envelope & synthesis

| Parameter | Field | Default | Meaning |
|-----------|-------|---------|---------|
| Envelope section BW | `envelope_section_bandwidth_hz` | 350 Hz | Piecewise-linear control point spacing |
| Notch radius | `harmonic_notch_radius_bins` | 2 | Exclude partial bins from envelope fit |
| Grain length | `stochastic_grain_length` | 2048 | Hanning window length (samples) |
| OLA hop | `overlap_add_hop` | 512 | Overlap-add advance |
| Random seed | `random_seed` | 42 | Reproducible stochastic resynthesis |

### 6.5 Seminar-specific plot constants

| Constant | Value | Used for |
|----------|-------|----------|
| `ATTACK_CROP_S` | 0.18 s | Visual subtraction figure window |
| `TRAJECTORY_T_MAX_S` | 1.5 s | Trajectory plot time axis |
| `RESYNTH_PLOT_MAX_S` | 0.25 s | Waveform comparison panel |
| `STRETCH_FACTOR` | 3.0 | Time-stretch comparison figure |
| `F_MAX_DIAG_HZ` | 4000 Hz | Diagnostic spectrogram cap |

---

## 7. Demo audio & expected behavior

### Clean C3 monophonic guitar (`audio/clean_guitar_slide.wav`)

| Quantity | Typical value | Notes |
|----------|---------------|-------|
| Partial count \(k\) | ~200 | Peak continuation, not harmonic grid |
| Estimated \(f_0\) | ~130–131 Hz | Parabolic refinement from ~129 Hz bin |
| Deterministic branch | Sustained harmonics, weak attack | Tones without pick transient |
| Stochastic branch | Broadband pick burst + friction | Magnitude-subtracted residual |
| Resynthesis quality | Perceptually close | Phase discarded on residual → not bit-perfect |

### High-gain Gojira chord (`audio/gojira_chord.wav`)

| Quantity | Typical value | Notes |
|----------|---------------|-------|
| Partial count \(k\) | ~900+ | Dense spectrum, many spurious guides |
| Failure mode | Superposition collapse | Intermodulation sidebands, no clean ridges |
| Audible result | Thin, fizzy, robotic | Linear det+stoch model breaks down |
| Comparison | Neural amp model | Non-linear; SMS cannot model saturation |

---

## 8. Musical control (why SMS matters)

Because \(d(t)\) and \(e(t)\) are **independently parameterized**:

| Operation | Deterministic branch | Stochastic branch | Benefit |
|-----------|---------------------|-------------------|---------|
| **Time-stretch** | Resample partial trajectories | Stretch tail; **lock attack duration** | Sharp transients preserved (vs phase vocoder smear) |
| **Pitch-shift** | Scale partial frequencies | Keep noise envelope fixed | No “chipmunk” breath/noise |
| **Cross-synthesis** | Source A partials | Source B residual envelope | Hybrid timbres (e.g. flute breath on guitar) |

**Time-stretch figure** (`scripts/figures/generate_figure_time_stretch_comparison.py`):

- 3× stretch factor.
- Naive phase vocoder: uniform stretch + Gaussian attack smear → soft swell.
- SMS: stretch det + stochastic tail; **first 2 ms of stochastic attack unchanged**.

---

## 9. Limitations (superposition collapse)

SMS assumes **linear additivity**: \(s(t) = d(t) + e(t)\).

**Breaks when:**

1. **Non-linear processing** (tube saturation, heavy distortion, neural amp modeling).
2. **Intermodulation** creates partials that are **not** integer harmonics of a single \(f_0\).
3. Peak tracker assigns energy to wrong guides or spawns hundreds of unstable tracks.
4. Magnitude subtraction leaves **complex residue** that is neither tonal nor simple noise.

**Clean power chord:** distinct harmonic ridges → tracking works.  
**Distorted chord:** chaotic dense spectrum → tracking fails → poor resynthesis.

This is a **model limitation**, not merely a parameter tuning issue.

---

## 10. SMS vs DDSP (conclusion slide)

| Aspect | SMS (1990) | DDSP (2020) |
|--------|------------|-------------|
| Analysis | Heuristic peak tracking + magnitude subtraction | Neural network predicts parameters |
| Synthesizer | Additive oscillators + filtered noise OLA | **Same** differentiable synthesizer |
| Role today | Foundational decomposition | SMS as **inductive bias** in learned models |
| Weakness | Polyphony, distortion, tracking errors | Requires training data |

SMS is not legacy code — it defines the **signal representation** modern systems still use.

---

## 11. Likely Q&A — quick answers

**Q: Why split deterministic and stochastic instead of using only sinusoids?**  
A: Wideband transients (pick attacks) need energy at many frequencies simultaneously. Representing that with sinusoids is inefficient (many partials, poor fit). Filtered noise captures the residual cheaply.

**Q: Why magnitude subtraction instead of subtracting the time-domain deterministic signal?**  
A: The stochastic model is **spectral-envelope based** with **random phase**. Working in magnitude domain matches the incoherent noise assumption and avoids fragile phase cancellation.

**Q: Why Blackman–Harris for analysis but Hanning for stochastic grains?**  
A: Analysis needs **low sidelobes** for accurate peak picking. Synthesis grains need **smooth OLA** with good overlap-add properties; Hanning tapers grains to zero at edges.

**Q: What does “80 Hz deviation” mean?**  
A: When linking peaks frame-to-frame, a guide accepts only peaks within ±80 Hz of its predicted frequency. Too small → tracks break; too large → wrong peak assignments.

**Q: What is “sleeping time = 3 frames”?**  
A: If a guide finds no matching peak for 3 consecutive frames (~35 ms), it is terminated. Prevents dangling tracks on noise.

**Q: Why ~200 partials for a monophonic guitar?**  
A: SMS tracks **every strong spectral peak**, including harmonics, formants, and beating sidebands — not just \(n \cdot f_0\) for fixed \(n\).

**Q: Why does resynthesis sound good but not identical?**  
A: Deterministic phase is coherent; stochastic phase is **randomized** each frame. Magnitude envelope is preserved; fine phase structure of noise is not.

**Q: Can SMS handle polyphony?**  
A: Poorly. Multiple simultaneous \(f_0\) contours confuse harmonic association and peak continuation. Works best on monophonic or sparse signals.

**Q: What happens under distortion?**  
A: Non-linear mixing creates new partials (intermodulation). The linear \(d + e\) model cannot separate them. Tracking and subtraction both fail → thin/fizzy output.

**Q: What is \(\Delta f\) for our STFT?**  
A: \(f_s / N_\mathrm{FFT} = 44100 / 2048 \approx 21.53\,\mathrm{Hz}\).

**Q: How much overlap?**  
A: Hop 512 with window 2048 → overlap \((2048-512)/2048 = 75\%\).

**Q: What parameters would you tune first?**  
A: (1) `peak_detection_threshold_db` — more/fewer peaks; (2) `max_partial_frequency_deviation_hz` — tracking stability; (3) `max_consecutive_missed_frames` — track lifetime; (4) analysis window type (Blackman–Harris vs Hann/Kaiser in test scripts).

---

## 12. Key equations (exam-style summary)

**Decomposition model:**
\[
s(t) = \sum_{r=1}^{k} A_r(t)\cos[\theta_r(t)] + e(t)
\]

**Residual magnitude:**
\[
|E_\ell(k)| = \bigl||X_\ell(k)| - |D_\ell(k)|\bigr|
\]

**Parabolic sub-bin offset:**
\[
p = \frac{\alpha - \gamma}{2(\alpha - 2\beta + \gamma)}
\]

**Phase accumulation (trapezoidal):**
\[
\theta[n] = \theta[n-1] + \frac{\pi(f[n] + f[n-1])}{f_s}
\]

**Frequency resolution:**
\[
\Delta f = \frac{f_s}{N_\mathrm{FFT}}
\]

---

## 13. Reproducibility commands

From the repository root:

```bash
pip install -r requirements.txt
python examples/run_sms.py path/to/mono.wav --output-dir output/
python examples/plot_decomposition.py path/to/mono.wav --output output/decomposition.png
python tests/test_parabola.py path/to/mono.wav
python tests/compare_analysis_windows.py --input path/to/mono.wav
python tests/run_parameter_sweep.py --input path/to/mono.wav
```

**Output assets:**

| Type | Path |
|------|------|
| Clean det/stoch/full | `audio/sms_*_clean_slide.wav` |
| Gojira det/stoch/full | `audio/sms_*_gojira_chord.wav` |
| Decomposition spectrogram | `figures/plot_intro_decomposition_spectra.pdf` |
| Trajectories | `figures/plot_analysis_det_trajectories.pdf` |
| Visual subtraction | `figures/plot_analysis_visual_subtraction.pdf` |
| Parabolic interpolation | `figures/python_parabolic_interpolation_v2.pdf` |
| Time-stretch comparison | `figures/python_time_stretch_comparison.pdf` |
| Superposition collapse | `figures/plot3_superposition_collapse.pdf` |

---

## 14. File map (code)

```
config/
└── sms_defaults.yaml          # Paper-default parameters

sms/
├── config.py                  # SmsParameters + YAML loader
├── analysis.py                # STFT, peaks, parabolic fit, continuation
├── envelope.py                # Magnitude subtraction envelope (Fig. 7)
├── synthesis.py               # Full SMS pipeline
├── audio_io.py                # Mono WAV load/export
└── time_stretch.py            # Time-stretch demo helpers

examples/
├── run_sms.py                 # Primary CLI entry point
└── plot_decomposition.py      # Optional decomposition figure

tests/
├── test_parabola.py           # Parabolic interpolation check
├── compare_analysis_windows.py
└── run_parameter_sweep.py
```

---

## 15. References

1. X. Serra & J. O. Smith, “Spectral Modeling Synthesis of Sounds,” *Computer Music Journal*, 14(4), 1990.
2. J. Bonada et al., “Spectral approach to the modeling of the singing voice,” *Proc. 111th AES Convention*, 2001. (Synthesis architecture diagram)
3. J. Engel et al., “DDSP: Differentiable Digital Signal Processing,” ICLR 2020. (Modern learned-parameter counterpart)

---

*Technical reference for the Spectral Modeling Synthesis student project — aligns with the faithful SMS implementation in this repository.*
