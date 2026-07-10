# Spectral Modeling Synthesis

A Python implementation of **Spectral Modeling Synthesis (SMS)** from Serra & Smith (1990). The signal is decomposed into a **deterministic** branch (tracked sinusoidal partials) and a **stochastic** branch (filtered noise residual):

\[
s(t) = d(t) + e(t) = \sum_{r} A_r(t)\cos[\theta_r(t)] + e(t)
\]

This repository was developed as a **student project** for the Audio Seminar (SS26) during my MSc studies in **Communication and Multimedia Engineering (CME)** at **Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)**.

## About

SMS separates tonal structure from broadband energy — harmonics are tracked across time, while the residual (pick attacks, breath, friction) is modeled as spectrally shaped noise. The implementation follows the original paper closely: Blackman–Harris STFT analysis, log-magnitude peak detection, parabolic interpolation, greedy peak continuation, magnitude subtraction, and Hanning overlap-add stochastic grains.

This is an **educational project** for learning and experimentation. It is not intended as a production audio tool.

## Paper reference

X. Serra & J. O. Smith, ["Spectral Modeling Synthesis of Sounds"](https://ccrma.stanford.edu/~jos/sasp/serra_smith_sms.html), *Computer Music Journal*, 14(4), 1990.

## Quick start

```bash
pip install -r requirements.txt
python examples/run_sms.py path/to/your_mono.wav --output-dir output/
```

**Outputs** (written to `output/` by default):

| File | Content |
|------|---------|
| `*_deterministic.wav` | Sinusoidal branch \(d(t)\) |
| `*_stochastic.wav` | Residual branch \(e(t)\) |
| `*_resynthesis.wav` | Full reconstruction \(d(t) + e(t)\) |

Provide your own **mono WAV** input. The repository does not bundle audio files.

Optional decomposition figure:

```bash
python examples/plot_decomposition.py path/to/your_mono.wav
```

## Configuration

Default parameters from the paper live in [`config/sms_defaults.yaml`](config/sms_defaults.yaml):

| Parameter | Default | Paper meaning |
|-----------|---------|---------------|
| `analysis.n_fft` | 2048 | STFT window length |
| `analysis.hop_size` | 512 | 75% overlap (\(M = 4H\)) |
| `analysis.window` | `blackman_harris` | Low-sidelobe analysis window |
| `peak_detection.threshold_db` | −50 | Log-magnitude peak threshold |
| `continuation.max_frequency_deviation_hz` | 80 | Peak–guide matching radius |
| `continuation.max_missed_frames` | 3 | Guide "sleeping time" |
| `stochastic.envelope_section_bandwidth_hz` | 350 | Piecewise-linear envelope (Fig. 7) |
| `stochastic.grain_length` | 2048 | Hanning grain length |
| `stochastic.overlap_add_hop` | 512 | OLA hop size |

Override with `--config path/to/custom.yaml` in the example scripts.

## Project structure

```
config/          Paper-default parameters (YAML)
sms/             Core library (analysis, synthesis, envelope)
examples/        CLI tools for running SMS on your audio
docs/            Technical reference (equations, Q&A)
tests/           Validation and parameter-sweep scripts
```

See [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) for equations, pipeline stages, and parameter details.

## How it works (brief)

1. **Analysis (deterministic):** Compute magnitude STFT → detect peaks in log-magnitude → parabolic sub-bin refinement → link peaks across frames with frequency guides.
2. **Synthesis (deterministic):** Resynthesize partials as additive oscillators with trapezoidal phase integration.
3. **Analysis (stochastic):** Subtract deterministic STFT magnitude from original: \(|E_\ell| = ||X_\ell| - |D_\ell||\).
4. **Synthesis (stochastic):** Fit piecewise-linear spectral envelope → random-phase grains → overlap-add.

## Limitations

- Assumes **linear superposition** — breaks down under heavy distortion or strong non-linear processing.
- Works best on **monophonic** or sparse signals; polyphony confuses peak continuation.
- Resynthesis is **not bit-perfect** — stochastic phase is randomized each frame (as in the original method).

## License

MIT License — see [LICENSE](LICENSE).

## Disclaimer

This software is provided for educational purposes as part of a university seminar project. Use at your own discretion.
