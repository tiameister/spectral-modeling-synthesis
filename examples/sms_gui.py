#!/usr/bin/env python3
"""
Tabbed GUI for Spectral Modeling Synthesis.

Tabs:
    General  — load audio, parameters, run synthesis, log
    Visuals  — waveform previews for all branches
    Sounds   — playback and export
    Figures  — optional analysis plots (see examples/plot_decomposition.py)

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

from examples.sms_visuals import create_decomposition_figure, waveform_envelope
from sms import DEFAULT_SMS_PARAMS, export_wav, load_mono_audio, spectral_modeling_synthesis

OUTPUT_DIR = ROOT / "output" / "gui"
FIGURES_DIR = OUTPUT_DIR / "figures"
MAX_LOG_LINES = 80
FIGURES_TAB = "Figures"

FIGURE_OPTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "decomposition",
        "Decomposition spectrograms",
        "examples/plot_decomposition.py",
    ),
)


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


WAVEFORM_BRANCHES: tuple[tuple[str, str, str, str], ...] = (
    ("original", "Original", Theme.WAVEFORM, Theme.WAVEFORM_FILL),
    ("det", "Deterministic", Theme.ORANGE, "#3d2a10"),
    ("stoch", "Stochastic", Theme.MAGENTA, "#2d1a3d"),
    ("full", "Resynthesis", Theme.TEAL, "#0d2d30"),
)

PLAYBACK_BRANCHES: tuple[tuple[str, str, str], ...] = (
    ("original", "Original", Theme.WAVEFORM),
    ("det", "Deterministic", Theme.ORANGE),
    ("stoch", "Stochastic", Theme.MAGENTA),
    ("full", "Resynthesis", Theme.TEAL),
)


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


class WaveformPanel(ctk.CTkFrame):
    """Compact labeled waveform strip."""

    def __init__(self, parent: ctk.CTkFrame, title: str, line_color: str, fill_color: str) -> None:
        super().__init__(parent, fg_color=Theme.SURFACE_ALT, corner_radius=10)
        self.line_color = line_color
        self.fill_color = fill_color
        self._cache: list[float] = []

        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=line_color,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))

        self.canvas = tk.Canvas(self, height=64, bg=Theme.SURFACE_ALT, highlightthickness=0, bd=0)
        self.canvas.grid(row=1, column=0, sticky="ew", padx=10, pady=(6, 10))
        self.canvas.bind("<Configure>", self._on_resize)
        self._draw_placeholder()

    def set_audio(self, audio: np.ndarray | None) -> None:
        self._cache = waveform_envelope(audio)
        self._redraw()

    def clear(self) -> None:
        self._cache = []
        self._draw_placeholder()

    def _draw_placeholder(self) -> None:
        self.canvas.delete("all")
        w = max(self.canvas.winfo_width(), 200)
        h = 64
        mid = h // 2
        self.canvas.create_line(0, mid, w, mid, fill=Theme.BORDER, width=1)
        self.canvas.create_text(w // 2, mid, text="—", fill=Theme.MUTED, font=("Segoe UI", 10))

    def _on_resize(self, _event: tk.Event) -> None:
        self._redraw()

    def _redraw(self) -> None:
        self.canvas.delete("all")
        w = max(self.canvas.winfo_width(), 200)
        h = 64
        mid = h / 2
        amp = (h / 2) - 6

        if not self._cache:
            self._draw_placeholder()
            return

        points: list[float] = []
        n = len(self._cache)
        for i, sample in enumerate(self._cache):
            x = i * (w - 1) / max(n - 1, 1)
            y = mid - sample * amp
            points.extend((x, y))

        fill_points = [(0, mid), *zip(points[::2], points[1::2]), (w, mid)]
        flat_fill = [coord for point in fill_points for coord in point]
        self.canvas.create_polygon(*flat_fill, fill=self.fill_color, outline="")
        self.canvas.create_line(*points, fill=self.line_color, width=1.5, smooth=True)
        self.canvas.create_line(0, mid, w, mid, fill=Theme.BORDER, width=1)


class SmsGui(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("SMS Studio")
        self.geometry("760x780")
        self.minsize(680, 640)
        self.configure(fg_color=Theme.BG)

        self.input_path: Path | None = None
        self.sample_rate_hz = 44100
        self.original: np.ndarray | None = None
        self.det: np.ndarray | None = None
        self.stoch: np.ndarray | None = None
        self.full: np.ndarray | None = None
        self.analysis = None
        self.analysis_info = ""
        self._playing_branch: str | None = None
        self._progress_job: str | None = None
        self._param_sync_guard = False
        self._figures_tab_visible = False
        self._figure_canvas = None
        self._figure_toolbar = None
        self._current_figure = None

        self.param_widgets: dict[str, Any] = {}
        self.param_vars: dict[str, tk.StringVar] = {}
        self.play_buttons: dict[str, ctk.CTkButton] = {}
        self.wave_panels: dict[str, WaveformPanel] = {}
        self.figure_vars: dict[str, tk.BooleanVar] = {}

        self._build_ui()
        self._bind_shortcuts()
        self.report_callback_exception = self._report_callback_exception  # type: ignore[method-assign]
        self._set_status("idle", "Load a mono WAV on the General tab to begin.")
        self._update_playback_state()

    def _report_callback_exception(
        self, exc: type[BaseException], value: BaseException, _tb
    ) -> None:
        messagebox.showerror("Unexpected error", f"{exc.__name__}: {value}")
        self._log(f"Unexpected error: {exc.__name__}: {value}")

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()

        self.tabs = ctk.CTkTabview(
            self,
            fg_color=Theme.SURFACE,
            segmented_button_fg_color=Theme.SURFACE_ALT,
            segmented_button_selected_color=Theme.ORANGE,
            segmented_button_selected_hover_color=Theme.ORANGE_DIM,
            segmented_button_unselected_color=Theme.SURFACE_ALT,
            segmented_button_unselected_hover_color=Theme.BORDER,
            text_color=Theme.TEXT,
        )
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 16))

        self.tab_general = self.tabs.add("General")
        self.tab_visuals = self.tabs.add("Visuals")
        self.tab_sounds = self.tabs.add("Sounds")

        for tab in (self.tab_general, self.tab_visuals, self.tab_sounds):
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)

        self._build_general_tab()
        self._build_visuals_tab()
        self._build_sounds_tab()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 8))
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

    def _card(self, parent: ctk.CTkFrame, title: str | None = None) -> ctk.CTkFrame:
        outer = ctk.CTkFrame(parent, fg_color="transparent")
        outer.grid_columnconfigure(0, weight=1)
        if title:
            ctk.CTkLabel(
                outer,
                text=title.upper(),
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=Theme.MUTED,
                anchor="w",
            ).grid(row=0, column=0, sticky="w", pady=(0, 6))
            row = 1
        else:
            row = 0

        card = ctk.CTkFrame(
            outer, fg_color=Theme.SURFACE_ALT, corner_radius=12, border_width=1, border_color=Theme.BORDER
        )
        card.grid(row=row, column=0, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        outer.card = card  # type: ignore[attr-defined]
        return outer

    def _build_general_tab(self) -> None:
        scroll = ctk.CTkScrollableFrame(self.tab_general, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        scroll.grid_columnconfigure(0, weight=1)

        # Source audio
        source_wrap = self._card(scroll, "Source audio")
        source_wrap.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        source = source_wrap.card  # type: ignore[attr-defined]

        info = ctk.CTkFrame(source, fg_color="transparent")
        info.pack(fill="x", padx=14, pady=14)
        info.grid_columnconfigure(0, weight=1)

        self.path_var = tk.StringVar(value="No file loaded")
        ctk.CTkLabel(
            info,
            textvariable=self.path_var,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=Theme.TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.meta_var = tk.StringVar(value="Load a mono WAV to start.")
        ctk.CTkLabel(
            info,
            textvariable=self.meta_var,
            font=ctk.CTkFont(size=12),
            text_color=Theme.MUTED,
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        actions = ctk.CTkFrame(info, fg_color="transparent")
        actions.grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 0))

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
            fg_color=Theme.SURFACE,
            hover_color=Theme.BORDER,
            border_width=1,
            border_color=Theme.BORDER,
            command=self._open_output_dir,
        ).pack(side="left")

        # Parameters
        params_wrap = self._card(scroll, "Parameters")
        params_wrap.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        params_card = params_wrap.card  # type: ignore[attr-defined]

        params_scroll = ctk.CTkScrollableFrame(params_card, fg_color="transparent", height=200)
        params_scroll.pack(fill="x", padx=8, pady=8)
        params_scroll.grid_columnconfigure((0, 1), weight=1, uniform="params")

        for index, spec in enumerate(PARAM_SPECS):
            self._add_param_widget(params_scroll, spec, index // 2, index % 2)

        ctk.CTkButton(
            params_card,
            text="Reset defaults",
            width=120,
            height=28,
            fg_color="transparent",
            hover_color=Theme.SURFACE,
            border_width=1,
            border_color=Theme.BORDER,
            text_color=Theme.MUTED,
            command=self._reset_defaults,
        ).pack(anchor="w", padx=14, pady=(0, 12))

        # Synthesis
        run_wrap = self._card(scroll, "Synthesis")
        run_wrap.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        run_card = run_wrap.card  # type: ignore[attr-defined]

        run_inner = ctk.CTkFrame(run_card, fg_color="transparent")
        run_inner.pack(fill="x", padx=14, pady=14)
        run_inner.grid_columnconfigure(0, weight=1)

        self.run_btn = ctk.CTkButton(
            run_inner,
            text="Run SMS",
            height=42,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=Theme.ORANGE,
            hover_color=Theme.ORANGE_DIM,
            text_color="#1a1000",
            command=self._run_sms,
        )
        self.run_btn.grid(row=0, column=0, sticky="ew")

        status_row = ctk.CTkFrame(run_inner, fg_color="transparent")
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

        self.progress = ctk.CTkProgressBar(run_inner, height=6, progress_color=Theme.TEAL, fg_color=Theme.BORDER)
        self.progress.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.progress.set(0)
        self.progress.grid_remove()

        # Log
        log_wrap = self._card(scroll, "Log")
        log_wrap.grid(row=3, column=0, sticky="ew")
        log_card = log_wrap.card  # type: ignore[attr-defined]

        self.log = ctk.CTkTextbox(
            log_card,
            height=120,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=Theme.SURFACE,
            text_color=Theme.MUTED,
            wrap="word",
        )
        self.log.pack(fill="x", padx=10, pady=10)
        self.log.configure(state="disabled")

    def _build_visuals_tab(self) -> None:
        scroll = ctk.CTkScrollableFrame(self.tab_visuals, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        scroll.grid_columnconfigure(0, weight=1)

        hint = ctk.CTkLabel(
            scroll,
            text="Waveform previews update after loading audio and running SMS.",
            font=ctk.CTkFont(size=12),
            text_color=Theme.MUTED,
            anchor="w",
        )
        hint.grid(row=0, column=0, sticky="w", pady=(0, 10))

        for row, (key, title, color, fill) in enumerate(WAVEFORM_BRANCHES):
            panel = WaveformPanel(scroll, title, color, fill)
            panel.grid(row=row + 1, column=0, sticky="ew", pady=4)
            self.wave_panels[key] = panel

        options_wrap = self._card(scroll, "Analysis figures")
        options_wrap.grid(row=5, column=0, sticky="ew", pady=(16, 0))
        options = options_wrap.card  # type: ignore[attr-defined]

        options_inner = ctk.CTkFrame(options, fg_color="transparent")
        options_inner.pack(fill="x", padx=14, pady=14)

        self.show_figures_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            options_inner,
            text="Enable Figures tab",
            variable=self.show_figures_var,
            checkbox_height=20,
            checkbox_width=20,
            fg_color=Theme.TEAL,
            hover_color=Theme.TEAL_DIM,
            border_color=Theme.BORDER,
            command=self._toggle_figures_tab,
        ).pack(anchor="w")

        ctk.CTkLabel(
            options_inner,
            text="Adds a Figures tab with spectrogram plots from examples/plot_decomposition.py",
            font=ctk.CTkFont(size=11),
            text_color=Theme.MUTED,
            anchor="w",
            wraplength=560,
        ).pack(anchor="w", pady=(8, 12))

        for key, label, script in FIGURE_OPTIONS:
            var = tk.BooleanVar(value=True)
            self.figure_vars[key] = var
            row = ctk.CTkFrame(options_inner, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkCheckBox(
                row,
                text=label,
                variable=var,
                checkbox_height=18,
                checkbox_width=18,
                fg_color=Theme.ORANGE,
                hover_color=Theme.ORANGE_DIM,
                border_color=Theme.BORDER,
            ).pack(side="left")
            ctk.CTkLabel(
                row,
                text=f"  ({script})",
                font=ctk.CTkFont(size=11),
                text_color=Theme.MUTED,
            ).pack(side="left")

    def _build_sounds_tab(self) -> None:
        scroll = ctk.CTkScrollableFrame(self.tab_sounds, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        scroll.grid_columnconfigure(0, weight=1)

        play_wrap = self._card(scroll, "Playback")
        play_wrap.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        play_card = play_wrap.card  # type: ignore[attr-defined]

        inner = ctk.CTkFrame(play_card, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=14)
        inner.grid_columnconfigure(0, weight=1)

        for row, (branch, label, color) in enumerate(PLAYBACK_BRANCHES):
            row_frame = ctk.CTkFrame(inner, fg_color=Theme.SURFACE, corner_radius=10)
            row_frame.grid(row=row, column=0, sticky="ew", pady=4)
            row_frame.grid_columnconfigure(1, weight=1)

            dot = ctk.CTkFrame(row_frame, width=4, height=32, fg_color=color, corner_radius=2)
            dot.grid(row=0, column=0, padx=(12, 10), pady=12)

            ctk.CTkLabel(
                row_frame,
                text=label,
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=Theme.TEXT,
                anchor="w",
            ).grid(row=0, column=1, sticky="w")

            btn = ctk.CTkButton(
                row_frame,
                text="Play",
                width=80,
                height=32,
                fg_color=Theme.SURFACE_ALT,
                hover_color=Theme.BORDER,
                border_width=1,
                border_color=Theme.BORDER,
                command=lambda b=branch: self._play(b),
            )
            btn.grid(row=0, column=2, padx=12, pady=12)
            self.play_buttons[branch] = btn

        action_row = ctk.CTkFrame(inner, fg_color="transparent")
        action_row.grid(row=4, column=0, sticky="ew", pady=(12, 0))

        self.stop_btn = ctk.CTkButton(
            action_row,
            text="Stop",
            width=100,
            height=36,
            fg_color=Theme.SURFACE,
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
            height=36,
            fg_color=Theme.TEAL,
            hover_color=Theme.TEAL_DIM,
            command=self._save_outputs,
        ).pack(side="right")

        export_wrap = self._card(scroll, "Analysis summary")
        export_wrap.grid(row=1, column=0, sticky="ew")
        export_card = export_wrap.card  # type: ignore[attr-defined]

        export_inner = ctk.CTkFrame(export_card, fg_color="transparent")
        export_inner.pack(fill="x", padx=14, pady=14)

        self.analysis_var = tk.StringVar(value="Run SMS to see f₀ and partial count.")
        ctk.CTkLabel(
            export_inner,
            textvariable=self.analysis_var,
            font=ctk.CTkFont(size=13),
            text_color=Theme.TEXT,
            anchor="w",
            wraplength=560,
        ).pack(anchor="w")

        ctk.CTkLabel(
            export_inner,
            text="Outputs are saved automatically to output/gui/ after each run.",
            font=ctk.CTkFont(size=11),
            text_color=Theme.MUTED,
            anchor="w",
        ).pack(anchor="w", pady=(8, 0))

    def _build_figures_tab(self) -> None:
        """Lazy-build the Figures tab content when first enabled."""
        self.tab_figures = self.tabs.add(FIGURES_TAB)
        self.tab_figures.grid_columnconfigure(0, weight=1)
        self.tab_figures.grid_rowconfigure(1, weight=1)

        toolbar = ctk.CTkFrame(self.tab_figures, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        toolbar.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            toolbar,
            text="Generate figures",
            width=140,
            height=34,
            fg_color=Theme.ORANGE,
            hover_color=Theme.ORANGE_DIM,
            text_color="#1a1000",
            font=ctk.CTkFont(weight="bold"),
            command=self._generate_figures,
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            toolbar,
            text="Save PNG…",
            width=110,
            height=34,
            fg_color=Theme.SURFACE_ALT,
            hover_color=Theme.BORDER,
            border_width=1,
            border_color=Theme.BORDER,
            command=self._save_figure_png,
        ).grid(row=0, column=1, padx=8)

        ctk.CTkButton(
            toolbar,
            text="Open figures folder",
            width=140,
            height=34,
            fg_color=Theme.SURFACE_ALT,
            hover_color=Theme.BORDER,
            border_width=1,
            border_color=Theme.BORDER,
            command=self._open_figures_dir,
        ).grid(row=0, column=2)

        self.figure_status_var = tk.StringVar(value="Run SMS, then generate figures.")
        ctk.CTkLabel(
            toolbar,
            textvariable=self.figure_status_var,
            font=ctk.CTkFont(size=11),
            text_color=Theme.MUTED,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        self.figure_host = ctk.CTkFrame(
            self.tab_figures, fg_color=Theme.SURFACE_ALT, corner_radius=12, border_width=1, border_color=Theme.BORDER
        )
        self.figure_host.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 8))
        self.figure_host.grid_columnconfigure(0, weight=1)
        self.figure_host.grid_rowconfigure(0, weight=1)

        self.figure_placeholder = ctk.CTkLabel(
            self.figure_host,
            text="No figures yet.\nEnable options on the Visuals tab, run SMS, then click Generate.",
            font=ctk.CTkFont(size=13),
            text_color=Theme.MUTED,
        )
        self.figure_placeholder.grid(row=0, column=0)

    def _toggle_figures_tab(self) -> None:
        if self.show_figures_var.get():
            if not self._figures_tab_visible:
                self._build_figures_tab()
                self._figures_tab_visible = True
            self.tabs.set(FIGURES_TAB)
        elif self._figures_tab_visible:
            self.tabs.delete(FIGURES_TAB)
            self._figures_tab_visible = False
            self._figure_canvas = None
            self._figure_toolbar = None
            self._current_figure = None

    def _add_param_widget(self, parent: ctk.CTkScrollableFrame, spec: ParamSpec, row: int, col: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color=Theme.SURFACE, corner_radius=8)
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
                fg_color=Theme.SURFACE_ALT,
                border_color=Theme.BORDER,
                button_color=Theme.BORDER,
                button_hover_color=Theme.MUTED,
                dropdown_fg_color=Theme.SURFACE_ALT,
                dropdown_hover_color=Theme.SURFACE,
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
                fg_color=Theme.SURFACE_ALT,
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
        self.bind("<Control-1>", lambda _e: self.tabs.set("General"))
        self.bind("<Control-2>", lambda _e: self.tabs.set("Visuals"))
        self.bind("<Control-3>", lambda _e: self.tabs.set("Sounds"))
        self.bind("<space>", self._on_stop_shortcut)
        self.bind("<Escape>", lambda _e: self._stop_playback())

    def _on_stop_shortcut(self, _event: tk.Event) -> str | None:
        focused = self.focus_get()
        if focused is not None and focused.winfo_class() in {"Entry", "Text"}:
            return None
        self._stop_playback()
        return "break"

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

    def _update_all_waveforms(self) -> None:
        audio_map = {
            "original": self.original,
            "det": self.det,
            "stoch": self.stoch,
            "full": self.full,
        }
        for key, panel in self.wave_panels.items():
            audio = audio_map.get(key)
            if audio is not None and len(audio) > 0:
                panel.set_audio(audio)
            elif key == "original":
                panel.clear()
            else:
                panel.clear()

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
            self.analysis = None
            self.analysis_info = ""
            self.analysis_var.set("Run SMS to see f₀ and partial count.")

            duration = len(audio) / sr
            self.path_var.set(self.input_path.name)
            self.meta_var.set(f"{duration:.2f} s  ·  {sr:,} Hz  ·  {len(audio):,} samples")
            self._update_all_waveforms()
            self._set_status("ok", f"Loaded {duration:.2f} s @ {sr} Hz")
            self._log(f"Loaded {self.input_path.name} ({duration:.2f} s, {sr} Hz)")
            self._update_playback_state()
            self.tabs.set("Visuals")
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
            self.analysis = analysis
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
            self.after(0, self._update_all_waveforms)
            self.after(0, self._update_playback_state)
            self.after(0, lambda: self.tabs.set("Visuals"))
            if self._figures_tab_visible and self.figure_vars["decomposition"].get():
                self.after(0, self._generate_figures)
        except Exception as exc:
            err = str(exc)
            self.after(0, lambda e=err: messagebox.showerror("SMS error", e))
            self.after(0, lambda: self._set_status("error", "Error during synthesis."))
            self.after(0, lambda e=err: self._log(f"Error: {e}"))
        finally:
            self.after(0, lambda: self._set_busy(False))

    # ── Figures ────────────────────────────────────────────────────────

    def _generate_figures(self) -> None:
        if not self._figures_tab_visible:
            messagebox.showinfo("Figures disabled", "Enable the Figures tab on the Visuals tab first.")
            return
        if self.det is None or self.stoch is None or self.analysis is None:
            messagebox.showinfo("No results", "Run SMS first.")
            return
        if not any(var.get() for var in self.figure_vars.values()):
            messagebox.showinfo("No figure selected", "Select at least one figure type on the Visuals tab.")
            return

        try:
            params = self._params_from_gui()
        except ValueError as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        want_decomposition = self.figure_vars["decomposition"].get()
        if hasattr(self, "figure_status_var"):
            self.figure_status_var.set("Generating figures…")
        threading.Thread(
            target=self._generate_figures_worker,
            args=(params, want_decomposition),
            daemon=True,
        ).start()

    def _generate_figures_worker(self, params, want_decomposition: bool) -> None:
        try:
            if want_decomposition:
                fig = create_decomposition_figure(
                    self.original,
                    self.det,
                    self.stoch,
                    self.sample_rate_hz,
                    self.analysis,
                    n_fft=params.n_fft,
                    hop_size=params.hop_size,
                )
                FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                stem = self.input_path.stem if self.input_path else "output"
                png_path = FIGURES_DIR / f"{stem}_decomposition.png"
                fig.savefig(png_path, dpi=120, bbox_inches="tight")
                self.after(0, lambda f=fig, p=png_path: self._show_figure(f, p))
            else:
                self.after(0, lambda: self.figure_status_var.set("No figure type selected."))
        except Exception as exc:
            err = str(exc)
            self.after(0, lambda e=err: messagebox.showerror("Figure error", e))
            self.after(0, lambda e=err: self.figure_status_var.set(f"Error: {e}"))

    def _show_figure(self, fig, path: Path) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        if self._figure_canvas is not None:
            self._figure_canvas.get_tk_widget().destroy()
        if self._figure_toolbar is not None:
            self._figure_toolbar.destroy()
        if self._current_figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self._current_figure)

        self.figure_placeholder.grid_remove()
        self._current_figure = fig
        self._figure_canvas = FigureCanvasTkAgg(fig, master=self.figure_host)
        self._figure_canvas.draw()
        self._figure_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.figure_status_var.set(f"Showing decomposition spectrograms — saved to {path}")
        self.tabs.set(FIGURES_TAB)

    def _save_figure_png(self) -> None:
        if self._current_figure is None:
            messagebox.showinfo("No figure", "Generate figures first.")
            return
        dest = filedialog.asksaveasfilename(
            title="Save figure",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if not dest:
            return
        self._current_figure.savefig(dest, dpi=150, bbox_inches="tight")
        self._log(f"Saved figure to {dest}")
        self.figure_status_var.set(f"Saved to {dest}")

    def _open_figures_dir(self) -> None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        import os

        os.startfile(FIGURES_DIR)  # type: ignore[attr-defined]

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
            btn.configure(state="normal" if enabled else "disabled", text="Play" if self._playing_branch != branch else "▶ Playing")
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
                fg_color=Theme.SURFACE,
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
