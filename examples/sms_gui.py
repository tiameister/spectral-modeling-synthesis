#!/usr/bin/env python3
"""
Simple GUI for Spectral Modeling Synthesis.

Load a mono WAV, adjust key SMS parameters, run synthesis, and play/save outputs.

Usage:
    python examples/sms_gui.py

Requires: customtkinter  (pip install customtkinter)
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Literal

import customtkinter as ctk
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sms import DEFAULT_SMS_PARAMS, export_wav, load_mono_audio, spectral_modeling_synthesis

OUTPUT_DIR = ROOT / "output" / "gui"
MAX_LOG_LINES = 80
WAVEFORM_POINTS = 180


# AudioLabs-inspired studio palette (beamerthemeal.sty)
class Theme:
    BG = "#0c0e14"
    SURFACE = "#141820"
    SURFACE_ALT = "#1c2230"
    BORDER = "#2a3142"
    ORANGE = "#ed8c01"
    ORANGE_DIM = "#b86d00"
    TEAL = "#0097a4"
    TEAL_DIM = "#007580"
    MAGENTA = "#d255ff"
    TEXT = "#eceff4"
    MUTED = "#7d8699"
    SUCCESS = "#2bae5b"
    ERROR = "#e85555"
    WAVEFORM = "#37bae2"
    WAVEFORM_FILL = "#1a3a4a"


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    kind: Literal["int", "float", "choice"]
    tooltip: str
    choices: tuple[str, ...] = ()
    min_val: float = 0.0
    max_val: float = 100.0
    step: float = 1.0

    def default_str(self) -> str:
        value = getattr(DEFAULT_SMS_PARAMS, self.attr_name)
        if self.kind == "choice":
            return str(value)
        if self.kind == "float":
            return f"{value:g}"
        return str(value)

    @property
    def attr_name(self) -> str:
        mapping = {
            "peak_threshold_db": "peak_detection_threshold_db",
            "max_freq_dev_hz": "max_partial_frequency_deviation_hz",
            "max_missed_frames": "max_consecutive_missed_frames",
            "envelope_bw_hz": "envelope_section_bandwidth_hz",
        }
        return mapping.get(self.key, self.key)


PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("n_fft", "FFT size", "choice", "STFT window length (samples).", ("512", "1024", "2048", "4096")),
    ParamSpec("hop_size", "Hop size", "choice", "Frame advance (samples).", ("128", "256", "512", "1024")),
    ParamSpec(
        "peak_threshold_db",
        "Peak threshold",
        "float",
        "Minimum peak level relative to frame max (dB).",
        min_val=-80.0,
        max_val=-20.0,
        step=1.0,
    ),
    ParamSpec(
        "max_freq_dev_hz",
        "Freq. deviation",
        "float",
        "Max partial frequency jump between frames (Hz).",
        min_val=20.0,
        max_val=200.0,
        step=5.0,
    ),
    ParamSpec(
        "max_missed_frames",
        "Missed frames",
        "int",
        "Frames a partial may disappear before termination.",
        min_val=1.0,
        max_val=10.0,
        step=1.0,
    ),
    ParamSpec(
        "envelope_bw_hz",
        "Envelope BW",
        "float",
        "Stochastic envelope section bandwidth (Hz).",
        min_val=100.0,
        max_val=800.0,
        step=25.0,
    ),
    ParamSpec(
        "random_seed",
        "Random seed",
        "int",
        "Seed for stochastic resynthesis noise.",
        min_val=0.0,
        max_val=9999.0,
        step=1.0,
    ),
)

PLAYBACK_BRANCHES: tuple[tuple[str, str, str], ...] = (
    ("original", "Original", Theme.WAVEFORM),
    ("det", "Deterministic", Theme.ORANGE),
    ("stoch", "Stochastic", Theme.MAGENTA),
    ("full", "Resynthesis", Theme.TEAL),
)


class SmsGui(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("SMS Studio")
        self.geometry("720x820")
        self.minsize(640, 700)
        self.configure(fg_color=Theme.BG)

        self.input_path: Path | None = None
        self.sample_rate_hz = 44100
        self.original: np.ndarray | None = None
        self.det: np.ndarray | None = None
        self.stoch: np.ndarray | None = None
        self.full: np.ndarray | None = None
        self.analysis_info = ""
        self._playing_branch: str | None = None
        self._waveform_cache: list[float] = []
        self._progress_job: str | None = None
        self._param_sync_guard = False

        self.param_widgets: dict[str, Any] = {}
        self.param_vars: dict[str, tk.StringVar] = {}
        self.play_buttons: dict[str, ctk.CTkButton] = {}

        self._build_ui()
        self._bind_shortcuts()
        self.report_callback_exception = self._report_callback_exception  # type: ignore[method-assign]
        self._set_status("idle", "Load a mono WAV to begin.")
        self._update_playback_state()

    def _report_callback_exception(
        self, exc: type[BaseException], value: BaseException, _tb
    ) -> None:
        messagebox.showerror("Unexpected error", f"{exc.__name__}: {value}")
        self._log(f"Unexpected error: {exc.__name__}: {value}")

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        self._build_header()
        self._build_file_section()
        self._build_params_section()
        self._build_run_section()
        self._build_playback_section()
        self._build_log_section()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 6))
        header.grid_columnconfigure(0, weight=1)

        title_row = ctk.CTkFrame(header, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew")
        title_row.grid_columnconfigure(1, weight=1)

        accent = ctk.CTkFrame(title_row, width=4, height=36, fg_color=Theme.ORANGE, corner_radius=2)
        accent.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 12))

        ctk.CTkLabel(
            title_row,
            text="SMS Studio",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=Theme.TEXT,
            anchor="w",
        ).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(
            title_row,
            text="Spectral Modeling Synthesis  ·  Serra & Smith, 1990",
            font=ctk.CTkFont(size=12),
            text_color=Theme.MUTED,
            anchor="w",
        ).grid(row=1, column=1, sticky="w", pady=(2, 0))

    def _section(self, row: int, title: str) -> ctk.CTkFrame:
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.grid(row=row, column=0, sticky="ew", padx=20, pady=(0, 10))
        outer.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            outer,
            text=title.upper(),
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=Theme.MUTED,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        card = ctk.CTkFrame(outer, fg_color=Theme.SURFACE, corner_radius=12, border_width=1, border_color=Theme.BORDER)
        card.grid(row=1, column=0, sticky="ew")
        card.grid_columnconfigure(0, weight=1)
        return card

    def _build_file_section(self) -> None:
        card = self._section(1, "Source audio")

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=14)
        inner.grid_columnconfigure(0, weight=1)

        self.wave_canvas = tk.Canvas(
            inner,
            height=72,
            bg=Theme.SURFACE_ALT,
            highlightthickness=0,
            bd=0,
        )
        self.wave_canvas.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.wave_canvas.bind("<Configure>", self._on_wave_resize)
        self._draw_empty_waveform()

        info = ctk.CTkFrame(inner, fg_color="transparent")
        info.grid(row=1, column=0, sticky="ew")
        info.grid_columnconfigure(0, weight=1)

        self.path_var = tk.StringVar(value="No file loaded")
        ctk.CTkLabel(
            info,
            textvariable=self.path_var,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=Theme.TEXT,
            anchor="w",
            wraplength=420,
        ).grid(row=0, column=0, sticky="w")

        self.meta_var = tk.StringVar(value="—")
        ctk.CTkLabel(
            info,
            textvariable=self.meta_var,
            font=ctk.CTkFont(size=12),
            text_color=Theme.MUTED,
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        actions = ctk.CTkFrame(inner, fg_color="transparent")
        actions.grid(row=1, column=1, sticky="e", padx=(12, 0))

        ctk.CTkButton(
            actions,
            text="Load WAV",
            width=110,
            height=34,
            fg_color=Theme.ORANGE,
            hover_color=Theme.ORANGE_DIM,
            text_color="#1a1000",
            font=ctk.CTkFont(weight="bold"),
            command=self._load_audio,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            actions,
            text="Output folder",
            width=110,
            height=34,
            fg_color=Theme.SURFACE_ALT,
            hover_color=Theme.BORDER,
            border_width=1,
            border_color=Theme.BORDER,
            command=self._open_output_dir,
        ).pack(side="left")

    def _build_params_section(self) -> None:
        card = self._section(2, "Parameters")

        scroll = ctk.CTkScrollableFrame(card, fg_color="transparent", height=220)
        scroll.pack(fill="x", padx=10, pady=10)
        scroll.grid_columnconfigure((0, 1), weight=1, uniform="params")

        for index, spec in enumerate(PARAM_SPECS):
            col = index % 2
            row = index // 2
            self._add_param_widget(scroll, spec, row, col)

        footer = ctk.CTkFrame(card, fg_color="transparent")
        footer.pack(fill="x", padx=14, pady=(0, 12))

        ctk.CTkButton(
            footer,
            text="Reset defaults",
            width=120,
            height=28,
            fg_color="transparent",
            hover_color=Theme.SURFACE_ALT,
            border_width=1,
            border_color=Theme.BORDER,
            text_color=Theme.MUTED,
            command=self._reset_defaults,
        ).pack(side="left")

    def _add_param_widget(self, parent: ctk.CTkScrollableFrame, spec: ParamSpec, row: int, col: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color=Theme.SURFACE_ALT, corner_radius=8)
        frame.grid(row=row, column=col, sticky="ew", padx=4, pady=4)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame,
            text=spec.label,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=Theme.TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 0))

        var = tk.StringVar(value=spec.default_str())
        self.param_vars[spec.key] = var

        if spec.kind == "choice":
            widget = ctk.CTkComboBox(
                frame,
                values=list(spec.choices),
                variable=var,
                height=30,
                fg_color=Theme.SURFACE,
                border_color=Theme.BORDER,
                button_color=Theme.BORDER,
                button_hover_color=Theme.MUTED,
                dropdown_fg_color=Theme.SURFACE,
                dropdown_hover_color=Theme.SURFACE_ALT,
            )
            widget.grid(row=1, column=0, sticky="ew", padx=10, pady=(4, 8))
        else:
            row_frame = ctk.CTkFrame(frame, fg_color="transparent")
            row_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(4, 8))
            row_frame.grid_columnconfigure(1, weight=1)

            entry = ctk.CTkEntry(
                row_frame,
                textvariable=var,
                width=72,
                height=30,
                fg_color=Theme.SURFACE,
                border_color=Theme.BORDER,
            )
            entry.grid(row=0, column=0, sticky="w")

            slider = ctk.CTkSlider(
                row_frame,
                from_=spec.min_val,
                to=spec.max_val,
                number_of_steps=max(1, int((spec.max_val - spec.min_val) / spec.step)),
                command=lambda v, k=spec.key: self._slider_to_var(k, v),
                progress_color=Theme.TEAL,
                button_color=Theme.TEAL,
                button_hover_color=Theme.TEAL_DIM,
                fg_color=Theme.BORDER,
            )
            slider.set(float(spec.default_str()))
            slider.grid(row=0, column=1, sticky="ew", padx=(8, 0))

            var.trace_add("write", lambda *_a, k=spec.key, s=slider: self._sync_slider(k, s))
            widget = (entry, slider)

        self.param_widgets[spec.key] = widget
        self._attach_tooltip(frame, spec.tooltip)

    def _build_run_section(self) -> None:
        card = self._section(3, "Synthesis")

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=14)
        inner.grid_columnconfigure(0, weight=1)

        self.run_btn = ctk.CTkButton(
            inner,
            text="Run SMS",
            height=42,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=Theme.ORANGE,
            hover_color=Theme.ORANGE_DIM,
            text_color="#1a1000",
            command=self._run_sms,
        )
        self.run_btn.grid(row=0, column=0, sticky="ew")

        status_row = ctk.CTkFrame(inner, fg_color="transparent")
        status_row.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        status_row.grid_columnconfigure(1, weight=1)

        self.status_dot = ctk.CTkLabel(status_row, text="●", font=ctk.CTkFont(size=14), text_color=Theme.MUTED)
        self.status_dot.grid(row=0, column=0, padx=(0, 8))

        self.status_var = tk.StringVar(value="")
        ctk.CTkLabel(
            status_row,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=12),
            text_color=Theme.TEXT,
            anchor="w",
            wraplength=520,
        ).grid(row=0, column=1, sticky="w")

        self.progress = ctk.CTkProgressBar(inner, height=6, progress_color=Theme.TEAL, fg_color=Theme.BORDER)
        self.progress.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.progress.set(0)
        self.progress.grid_remove()

    def _build_playback_section(self) -> None:
        card = self._section(4, "Playback")

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=14)
        inner.grid_columnconfigure(tuple(range(4)), weight=1, uniform="play")

        for col, (branch, label, color) in enumerate(PLAYBACK_BRANCHES):
            btn = ctk.CTkButton(
                inner,
                text=label,
                height=38,
                fg_color=Theme.SURFACE_ALT,
                hover_color=Theme.BORDER,
                border_width=1,
                border_color=Theme.BORDER,
                text_color=Theme.TEXT,
                command=lambda b=branch: self._play(b),
            )
            btn.grid(row=0, column=col, sticky="ew", padx=3)
            self.play_buttons[branch] = btn
            self._attach_tooltip(btn, f"Play {label.lower()} branch")

        action_row = ctk.CTkFrame(inner, fg_color="transparent")
        action_row.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(12, 0))
        action_row.grid_columnconfigure(0, weight=1)

        self.stop_btn = ctk.CTkButton(
            action_row,
            text="Stop",
            width=90,
            height=32,
            fg_color=Theme.SURFACE_ALT,
            hover_color=Theme.BORDER,
            border_width=1,
            border_color=Theme.BORDER,
            command=self._stop_playback,
        )
        self.stop_btn.pack(side="left")

        ctk.CTkButton(
            action_row,
            text="Save WAVs…",
            width=120,
            height=32,
            fg_color=Theme.TEAL,
            hover_color=Theme.TEAL_DIM,
            command=self._save_outputs,
        ).pack(side="right")

        self.analysis_var = tk.StringVar(value="")
        ctk.CTkLabel(
            action_row,
            textvariable=self.analysis_var,
            font=ctk.CTkFont(size=12),
            text_color=Theme.MUTED,
        ).pack(side="right", padx=(0, 16))

    def _build_log_section(self) -> None:
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.grid(row=5, column=0, sticky="nsew", padx=20, pady=(0, 16))
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(outer, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="LOG",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=Theme.MUTED,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.log_visible = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            header,
            text="Show",
            variable=self.log_visible,
            width=60,
            checkbox_height=18,
            checkbox_width=18,
            fg_color=Theme.TEAL,
            hover_color=Theme.TEAL_DIM,
            border_color=Theme.BORDER,
            command=self._toggle_log,
        ).grid(row=0, column=1, sticky="e")

        self.log_card = ctk.CTkFrame(outer, fg_color=Theme.SURFACE, corner_radius=12, border_width=1, border_color=Theme.BORDER)
        self.log_card.grid(row=1, column=0, sticky="nsew")
        self.log_card.grid_columnconfigure(0, weight=1)
        self.log_card.grid_rowconfigure(0, weight=1)

        self.log = ctk.CTkTextbox(
            self.log_card,
            height=110,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=Theme.SURFACE_ALT,
            text_color=Theme.MUTED,
            wrap="word",
            activate_scrollbars=True,
        )
        self.log.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.log.configure(state="disabled")

    # ── Small UI helpers ─────────────────────────────────────────────────

    def _attach_tooltip(self, widget: tk.Misc, text: str) -> None:
        tip: tk.Toplevel | None = None

        def show(_event: tk.Event) -> None:
            nonlocal tip
            if tip is not None:
                return
            x = widget.winfo_rootx() + 12
            y = widget.winfo_rooty() + widget.winfo_height() + 6
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                tip,
                text=text,
                bg=Theme.SURFACE_ALT,
                fg=Theme.TEXT,
                relief="solid",
                borderwidth=1,
                padx=8,
                pady=4,
                font=("Segoe UI", 9),
                wraplength=260,
            )
            label.pack()

        def hide(_event: tk.Event) -> None:
            nonlocal tip
            if tip is not None:
                tip.destroy()
                tip = None

        widget.bind("<Enter>", show, add="+")
        widget.bind("<Leave>", hide, add="+")

    def _slider_to_var(self, key: str, value: float) -> None:
        spec = next(s for s in PARAM_SPECS if s.key == key)
        self._param_sync_guard = True
        try:
            if spec.kind == "int":
                self.param_vars[key].set(str(int(round(value))))
            else:
                self.param_vars[key].set(f"{value:g}")
        finally:
            self._param_sync_guard = False

    def _sync_slider(self, key: str, slider: ctk.CTkSlider) -> None:
        if self._param_sync_guard:
            return
        raw = self.param_vars[key].get().strip()
        try:
            slider.set(float(raw))
        except ValueError:
            pass

    def _bind_shortcuts(self) -> None:
        self.bind("<Control-o>", lambda _e: self._load_audio())
        self.bind("<Control-r>", lambda _e: self._run_sms())
        self.bind("<space>", self._on_stop_shortcut)
        self.bind("<Escape>", lambda _e: self._stop_playback())

    def _on_stop_shortcut(self, _event: tk.Event) -> str | None:
        focused = self.focus_get()
        if focused is not None and focused.winfo_class() in {"Entry", "Text"}:
            return None
        self._stop_playback()
        return "break"

    def _toggle_log(self) -> None:
        if self.log_visible.get():
            self.log_card.grid()
        else:
            self.log_card.grid_remove()

    def _set_status(self, state: Literal["idle", "busy", "ok", "error"], message: str) -> None:
        colors = {
            "idle": Theme.MUTED,
            "busy": Theme.ORANGE,
            "ok": Theme.SUCCESS,
            "error": Theme.ERROR,
        }
        self.status_dot.configure(text_color=colors[state])
        self.status_var.set(message)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.run_btn.configure(state=state)
        for widget in self.param_widgets.values():
            if isinstance(widget, ctk.CTkComboBox):
                widget.configure(state=state)
            else:
                widget[0].configure(state=state)
                widget[1].configure(state=state)
        if busy:
            self.progress.grid()
            self._animate_progress()
            self._set_status("busy", "Running spectral analysis and resynthesis…")
        else:
            if self._progress_job is not None:
                self.after_cancel(self._progress_job)
            self.progress.grid_remove()
            self.progress.set(0)
            self._progress_job = None

    def _animate_progress(self) -> None:
        if self.run_btn.cget("state") == "disabled":
            current = self.progress.get()
            self.progress.set(0.0 if current >= 0.95 else current + 0.04)
            self._progress_job = self.after(80, self._animate_progress)

    # ── Waveform preview (lightweight) ───────────────────────────────────

    def _draw_empty_waveform(self) -> None:
        self.wave_canvas.delete("all")
        w = max(self.wave_canvas.winfo_width(), 400)
        h = 72
        mid = h // 2
        self.wave_canvas.create_line(0, mid, w, mid, fill=Theme.BORDER, width=1)
        self.wave_canvas.create_text(
            w // 2,
            mid,
            text="waveform preview",
            fill=Theme.MUTED,
            font=("Segoe UI", 10),
        )

    def _update_waveform(self, audio: np.ndarray | None) -> None:
        if audio is None or len(audio) == 0:
            self._waveform_cache = []
            self._draw_empty_waveform()
            return

        step = max(1, len(audio) // WAVEFORM_POINTS)
        chunk = audio[::step]
        peak = float(np.max(np.abs(chunk))) or 1.0
        self._waveform_cache = (chunk / peak).tolist()
        self._redraw_waveform()

    def _redraw_waveform(self) -> None:
        self.wave_canvas.delete("all")
        w = max(self.wave_canvas.winfo_width(), 400)
        h = 72
        mid = h / 2
        amp = (h / 2) - 6

        if not self._waveform_cache:
            self._draw_empty_waveform()
            return

        points: list[float] = []
        n = len(self._waveform_cache)
        for i, sample in enumerate(self._waveform_cache):
            x = i * (w - 1) / max(n - 1, 1)
            y = mid - sample * amp
            points.extend((x, y))

        fill_points = [(0, mid), *zip(points[::2], points[1::2]), (w, mid)]
        flat_fill = [coord for point in fill_points for coord in point]
        self.wave_canvas.create_polygon(*flat_fill, fill=Theme.WAVEFORM_FILL, outline="")
        self.wave_canvas.create_line(*points, fill=Theme.WAVEFORM, width=1.5, smooth=True)
        self.wave_canvas.create_line(0, mid, w, mid, fill=Theme.BORDER, width=1)

    def _on_wave_resize(self, _event: tk.Event) -> None:
        if self._waveform_cache:
            self._redraw_waveform()

    # ── Logging ──────────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        line_count = int(self.log.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            self.log.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ── Parameters ───────────────────────────────────────────────────────

    def _reset_defaults(self) -> None:
        for spec in PARAM_SPECS:
            self.param_vars[spec.key].set(spec.default_str())
        self._log("Parameters reset to defaults.")

    def _parse_int(self, key: str, label: str) -> int:
        raw = self.param_vars[key].get().strip()
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer (got {raw!r})") from exc

    def _parse_float(self, key: str, label: str) -> float:
        raw = self.param_vars[key].get().strip()
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number (got {raw!r})") from exc

    def _params_from_gui(self):
        return replace(
            DEFAULT_SMS_PARAMS,
            n_fft=self._parse_int("n_fft", "FFT size"),
            hop_size=self._parse_int("hop_size", "Hop size"),
            peak_detection_threshold_db=self._parse_float("peak_threshold_db", "Peak threshold"),
            max_partial_frequency_deviation_hz=self._parse_float("max_freq_dev_hz", "Freq. deviation"),
            max_consecutive_missed_frames=self._parse_int("max_missed_frames", "Missed frames"),
            envelope_section_bandwidth_hz=self._parse_float("envelope_bw_hz", "Envelope BW"),
            random_seed=self._parse_int("random_seed", "Random seed"),
        )

    # ── Audio I/O ────────────────────────────────────────────────────────

    def _load_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Select mono WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            audio, sr = load_mono_audio(path)
            if sr <= 0:
                raise ValueError(f"Invalid sample rate: {sr} Hz")
            if len(audio) == 0:
                raise ValueError("Audio file is empty.")
            if not np.isfinite(audio).all():
                raise ValueError("Audio contains invalid samples (NaN/Inf).")

            self.input_path = Path(path)
            self.original = audio
            self.sample_rate_hz = sr
            self.det = None
            self.stoch = None
            self.full = None
            self.analysis_info = ""
            self.analysis_var.set("")

            duration = len(audio) / sr
            self.path_var.set(self.input_path.name)
            self.meta_var.set(f"{duration:.2f} s  ·  {sr:,} Hz  ·  {len(audio):,} samples")
            self._update_waveform(audio)
            self._set_status("ok", f"Loaded {duration:.2f} s @ {sr} Hz")
            self._log(f"Loaded {self.input_path.name} ({duration:.2f} s, {sr} Hz)")
            self._update_playback_state()
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            self._log(f"Load error: {exc}")
            self._set_status("error", "Failed to load audio.")

    def _run_sms(self) -> None:
        if self.original is None:
            messagebox.showinfo("No audio", "Load a mono WAV file first.")
            return
        try:
            self._params_from_gui()
        except ValueError as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self._set_busy(True)
        threading.Thread(target=self._run_sms_worker, daemon=True).start()

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
                f"f₀ ≈ {analysis.f0_hz:.1f} Hz  ·  {len(analysis.partials)} partials"
            )

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            stem = self.input_path.stem if self.input_path else "output"
            export_wav(str(OUTPUT_DIR / f"{stem}_deterministic.wav"), det, self.sample_rate_hz)
            export_wav(str(OUTPUT_DIR / f"{stem}_stochastic.wav"), stoch, self.sample_rate_hz)
            export_wav(str(OUTPUT_DIR / f"{stem}_resynthesis.wav"), full, self.sample_rate_hz)

            msg = f"Done — {self.analysis_info}"
            self.after(0, lambda: self._set_status("ok", msg))
            self.after(0, lambda: self.analysis_var.set(self.analysis_info))
            self.after(0, lambda: self._log(msg))
            self.after(0, lambda: self._log(f"Saved outputs to {OUTPUT_DIR}"))
            self.after(0, self._update_playback_state)
        except Exception as exc:
            err = str(exc)
            self.after(0, lambda e=err: messagebox.showerror("SMS error", e))
            self.after(0, lambda: self._set_status("error", "Error during synthesis."))
            self.after(0, lambda e=err: self._log(f"Error: {e}"))
        finally:
            self.after(0, lambda: self._set_busy(False))

    # ── Playback ───────────────────────────────────────────────────────

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

    def _update_playback_state(self) -> None:
        has_audio = self.original is not None
        has_results = self.full is not None

        for branch, btn in self.play_buttons.items():
            if branch == "original":
                enabled = has_audio
            else:
                enabled = has_results
            btn.configure(state="normal" if enabled else "disabled")
            if not enabled and self._playing_branch == branch:
                self._playing_branch = None
            self._style_play_button(branch)

        self.stop_btn.configure(state="normal" if self._playing_branch else "disabled")

    def _style_play_button(self, branch: str) -> None:
        btn = self.play_buttons[branch]
        _, _, accent = next(item for item in PLAYBACK_BRANCHES if item[0] == branch)
        if self._playing_branch == branch:
            btn.configure(fg_color=accent, hover_color=accent, text_color="#0c0e14", border_color=accent)
        else:
            btn.configure(
                fg_color=Theme.SURFACE_ALT,
                hover_color=Theme.BORDER,
                text_color=Theme.TEXT,
                border_color=Theme.BORDER,
            )

    def _stop_playback(self, *, announce: bool = True) -> None:
        was_playing = self._playing_branch is not None
        try:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
        self._playing_branch = None
        self._update_playback_state()
        if announce and was_playing:
            self._log("Playback stopped.")

    def _play(self, branch: str) -> None:
        if branch != "original" and self.full is None:
            messagebox.showinfo("No results", "Run SMS first.")
            return
        path = self._wav_path(branch)
        if path is None or not path.is_file():
            messagebox.showinfo("Missing file", "Audio not available.")
            return

        self._stop_playback(announce=False)
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_ASYNC)
            self._playing_branch = branch
            self._update_playback_state()
            self._log(f"Playing {path.name}")
        except Exception as exc:
            try:
                import os

                os.startfile(path)  # type: ignore[attr-defined]
                self._log(f"Opened {path.name} with default app")
            except Exception as inner:
                messagebox.showerror("Playback error", f"{exc}\n{inner}")

    def _save_outputs(self) -> None:
        if self.full is None or self.det is None or self.stoch is None:
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
        self._set_status("ok", f"Exported WAVs to {dest}")

    def _open_output_dir(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        import os

        os.startfile(OUTPUT_DIR)  # type: ignore[attr-defined]


def main() -> None:
    app = SmsGui()
    app.mainloop()


if __name__ == "__main__":
    main()
