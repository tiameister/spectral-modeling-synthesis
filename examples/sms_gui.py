#!/usr/bin/env python3
"""
Simple GUI for Spectral Modeling Synthesis.

Load a mono WAV, adjust key SMS parameters, run synthesis, and play/save outputs.

Usage:
    python examples/sms_gui.py
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sms import DEFAULT_SMS_PARAMS, export_wav, load_mono_audio, spectral_modeling_synthesis

OUTPUT_DIR = ROOT / "output" / "gui"


class SmsGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spectral Modeling Synthesis")
        self.geometry("640x720")
        self.minsize(560, 640)

        self.input_path: Path | None = None
        self.sample_rate_hz = 44100
        self.original: np.ndarray | None = None
        self.det: np.ndarray | None = None
        self.stoch: np.ndarray | None = None
        self.full: np.ndarray | None = None
        self.analysis_info = ""

        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}

        file_frame = ttk.LabelFrame(self, text="Audio")
        file_frame.pack(fill="x", **pad)

        self.path_var = tk.StringVar(value="No file loaded")
        ttk.Label(file_frame, textvariable=self.path_var, wraplength=580).pack(
            anchor="w", padx=8, pady=6
        )

        btn_row = ttk.Frame(file_frame)
        btn_row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(btn_row, text="Load WAV…", command=self._load_audio).pack(side="left")
        ttk.Button(btn_row, text="Open output folder", command=self._open_output_dir).pack(
            side="left", padx=8
        )

        params_frame = ttk.LabelFrame(self, text="Parameters")
        params_frame.pack(fill="x", **padx)

        defaults = DEFAULT_SMS_PARAMS
        self.vars = {
            "n_fft": tk.IntVar(value=defaults.n_fft),
            "hop_size": tk.IntVar(value=defaults.hop_size),
            "peak_threshold_db": tk.DoubleVar(value=defaults.peak_detection_threshold_db),
            "max_freq_dev_hz": tk.DoubleVar(value=defaults.max_partial_frequency_deviation_hz),
            "max_missed_frames": tk.IntVar(value=defaults.max_consecutive_missed_frames),
            "envelope_bw_hz": tk.DoubleVar(value=defaults.envelope_section_bandwidth_hz),
            "random_seed": tk.IntVar(value=defaults.random_seed),
        }

        self._add_spin(params_frame, "FFT size (N)", "n_fft", 512, 8192, 256)
        self._add_spin(params_frame, "Hop size (H)", "hop_size", 128, 2048, 128)
        self._add_spin(params_frame, "Peak threshold (dB)", "peak_threshold_db", -80, -20, 1)
        self._add_spin(params_frame, "Freq. deviation (Hz)", "max_freq_dev_hz", 20, 200, 5)
        self._add_spin(params_frame, "Missed frames", "max_missed_frames", 0, 10, 1)
        self._add_spin(params_frame, "Envelope BW (Hz)", "envelope_bw_hz", 100, 800, 25)
        self._add_spin(params_frame, "Random seed", "random_seed", 0, 99999, 1)

        ttk.Button(params_frame, text="Reset to defaults", command=self._reset_defaults).pack(
            anchor="w", padx=8, pady=6
        )

        run_frame = ttk.Frame(self)
        run_frame.pack(fill="x", **pad)
        self.run_btn = ttk.Button(run_frame, text="Run SMS", command=self._run_sms)
        self.run_btn.pack(side="left")
        self.status_var = tk.StringVar(value="Load a mono WAV to begin.")
        ttk.Label(run_frame, textvariable=self.status_var, wraplength=480).pack(
            side="left", padx=12
        )

        play_frame = ttk.LabelFrame(self, text="Playback")
        play_frame.pack(fill="x", **pad)
        for label, cmd in [
            ("Original", lambda: self._play("original")),
            ("Deterministic", lambda: self._play("det")),
            ("Stochastic", lambda: self._play("stoch")),
            ("Resynthesis", lambda: self._play("full")),
        ]:
            ttk.Button(play_frame, text=label, command=cmd).pack(side="left", padx=6, pady=8)

        ttk.Button(play_frame, text="Save WAVs…", command=self._save_outputs).pack(
            side="left", padx=6, pady=8
        )

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

    def _add_spin(
        self,
        parent: ttk.LabelFrame,
        label: str,
        key: str,
        from_: float,
        to: float,
        step: float,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text=label, width=22).pack(side="left")
        if isinstance(step, float) or key.endswith("_db") or "hz" in key or "bw" in key:
            ttk.Spinbox(
                row,
                textvariable=self.vars[key],
                from_=from_,
                to=to,
                increment=step,
                width=10,
            ).pack(side="left")
        else:
            ttk.Spinbox(
                row,
                textvariable=self.vars[key],
                from_=int(from_),
                to=int(to),
                increment=int(step),
                width=10,
            ).pack(side="left")

    def _log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _reset_defaults(self) -> None:
        d = DEFAULT_SMS_PARAMS
        self.vars["n_fft"].set(d.n_fft)
        self.vars["hop_size"].set(d.hop_size)
        self.vars["peak_threshold_db"].set(d.peak_detection_threshold_db)
        self.vars["max_freq_dev_hz"].set(d.max_partial_frequency_deviation_hz)
        self.vars["max_missed_frames"].set(d.max_consecutive_missed_frames)
        self.vars["envelope_bw_hz"].set(d.envelope_section_bandwidth_hz)
        self.vars["random_seed"].set(d.random_seed)
        self._log("Parameters reset to paper defaults.")

    def _params_from_gui(self):
        return replace(
            DEFAULT_SMS_PARAMS,
            n_fft=int(self.vars["n_fft"].get()),
            hop_size=int(self.vars["hop_size"].get()),
            peak_detection_threshold_db=float(self.vars["peak_threshold_db"].get()),
            max_partial_frequency_deviation_hz=float(self.vars["max_freq_dev_hz"].get()),
            max_consecutive_missed_frames=int(self.vars["max_missed_frames"].get()),
            envelope_section_bandwidth_hz=float(self.vars["envelope_bw_hz"].get()),
            random_seed=int(self.vars["random_seed"].get()),
        )

    def _load_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Select mono WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            audio, sr = load_mono_audio(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        self.input_path = Path(path)
        self.original = audio
        self.sample_rate_hz = sr
        self.det = self.stoch = self.full = None
        self.path_var.set(str(self.input_path))
        duration = len(audio) / sr
        self.status_var.set(f"Loaded: {duration:.2f} s @ {sr} Hz")
        self._log(f"Loaded {self.input_path.name} ({duration:.2f} s, {sr} Hz)")

    def _run_sms(self) -> None:
        if self.original is None:
            messagebox.showinfo("No audio", "Load a mono WAV file first.")
            return
        self.run_btn.configure(state="disabled")
        self.status_var.set("Running SMS…")
        thread = threading.Thread(target=self._run_sms_worker, daemon=True)
        thread.start()

    def _run_sms_worker(self) -> None:
        try:
            params = self._params_from_gui()
            rng = np.random.default_rng(params.random_seed)
            det, stoch, full, analysis = spectral_modeling_synthesis(
                self.original,
                self.sample_rate_hz,
                rng=rng,
                params=params,
            )
            self.det, self.stoch, self.full = det, stoch, full
            self.analysis_info = (
                f"f0 ≈ {analysis.f0_hz:.1f} Hz, partials k = {len(analysis.partials)}"
            )

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            stem = self.input_path.stem if self.input_path else "output"
            export_wav(str(OUTPUT_DIR / f"{stem}_deterministic.wav"), det, self.sample_rate_hz)
            export_wav(str(OUTPUT_DIR / f"{stem}_stochastic.wav"), stoch, self.sample_rate_hz)
            export_wav(str(OUTPUT_DIR / f"{stem}_resynthesis.wav"), full, self.sample_rate_hz)

            msg = f"Done — {self.analysis_info}"
            self.after(0, lambda: self.status_var.set(msg))
            self.after(0, lambda: self._log(msg))
            self.after(0, lambda: self._log(f"Saved outputs to {OUTPUT_DIR}"))
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("SMS error", str(exc)))
            self.after(0, lambda: self.status_var.set("Error during synthesis."))
            self.after(0, lambda: self._log(f"Error: {exc}"))
        finally:
            self.after(0, lambda: self.run_btn.configure(state="normal"))

    def _wav_path(self, branch: str) -> Path | None:
        if self.input_path is None:
            return None
        stem = self.input_path.stem
        names = {
            "original": self.input_path,
            "det": OUTPUT_DIR / f"{stem}_deterministic.wav",
            "stoch": OUTPUT_DIR / f"{stem}_stochastic.wav",
            "full": OUTPUT_DIR / f"{stem}_resynthesis.wav",
        }
        return names.get(branch)

    def _play(self, branch: str) -> None:
        if branch != "original" and self.full is None:
            messagebox.showinfo("No results", "Run SMS first.")
            return
        path = self._wav_path(branch)
        if path is None or not path.is_file():
            messagebox.showinfo("Missing file", "Audio not available.")
            return
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_ASYNC)
            self._log(f"Playing {path.name}")
        except Exception:
            import os

            os.startfile(path)  # type: ignore[attr-defined]
            self._log(f"Opened {path.name} with default app")

    def _save_outputs(self) -> None:
        if self.full is None:
            messagebox.showinfo("No results", "Run SMS first.")
            return
        dest = filedialog.askdirectory(title="Choose output folder")
        if not dest:
            return
        stem = self.input_path.stem if self.input_path else "output"
        export_wav(str(Path(dest) / f"{stem}_deterministic.wav"), self.det, self.sample_rate_hz)
        export_wav(str(Path(dest) / f"{stem}_stochastic.wav"), self.stoch, self.sample_rate_hz)
        export_wav(str(Path(dest) / f"{stem}_resynthesis.wav"), self.full, self.sample_rate_hz)
        self._log(f"Saved WAVs to {dest}")

    def _open_output_dir(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        import os

        os.startfile(OUTPUT_DIR)  # type: ignore[attr-defined]


def main() -> None:
    app = SmsGui()
    app.mainloop()


if __name__ == "__main__":
    main()
