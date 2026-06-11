#!/usr/bin/env python3
"""
Response Analyzer — Advanced Signal Processing & Vibration Analysis Tool
========================================================================
A comprehensive desktop GUI for interactive response analysis of signal data.
Features: configurable digital filters (Butterworth + Moving Average), FFT
windowing, multi-peak detection with damping estimation, spectrogram (STFT),
Welch PSD, Hilbert envelope, multi-file comparison overlay, real-time signal
statistics, and CSV data export.

Author: Antigravity AI
Date: 2026-06-11
"""

import os
import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy import signal as sp_signal
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew

# ─── Theme Palette ──────────────────────────────────────────────────
DARK_BG      = "#12121a"
PANEL_BG     = "#1c1c2e"
CARD_BG      = "#252540"
TEXT_PRIMARY  = "#eaeaf0"
TEXT_MUTED    = "#7a7a99"
ACCENT_CYAN   = "#00d2d3"
ACCENT_RED    = "#ff6b6b"
ACCENT_GREEN  = "#1dd1a1"
ACCENT_AMBER  = "#feca57"
ACCENT_BLUE   = "#54a0ff"
GRID_CLR      = "#2e2e48"
BUTTON_BG     = "#33335a"
PLOT_BG       = "#16162a"

OVERLAY_COLORS = [
    "#ff6b6b", "#54a0ff", "#1dd1a1", "#feca57",
    "#a29bfe", "#fd79a8", "#00cec9", "#e17055",
]

FILTER_TYPES  = ["None", "Moving Average", "Lowpass", "Highpass", "Bandpass"]
WINDOW_FUNCS  = ["Rectangular", "Hanning", "Hamming", "Blackman", "Flat-top"]
VIEW_MODES    = ["Overview", "Time Domain", "Frequency Analysis", "Spectrogram", "PSD"]
SEGMENT_SIZES = ["64", "128", "256", "512", "1024"]

SCIPY_WIN_MAP = {
    "Rectangular": "boxcar", "Hanning": "hann", "Hamming": "hamming",
    "Blackman": "blackman", "Flat-top": "flattop",
}


class ResponseAnalyzer(tk.Tk):
    """Main application window."""

    # ═══════════════════════════════════════════════════════════
    #  INITIALIZATION
    # ═══════════════════════════════════════════════════════════
    def __init__(self):
        super().__init__()
        self.title("Response Analyzer — Signal & Vibration Analysis")
        self.geometry("1520x960")
        self.configure(bg=DARK_BG)

        # Resolve base directory (PyInstaller compatible)
        if getattr(sys, "frozen", False):
            self.base_dir = os.path.dirname(sys.executable)
        else:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # ── State ────────────────────────────────────────────
        self.signals = []       # list of loaded signal dicts
        self.primary_idx = 0    # which signal shows stats

        # ── Tk Variables ─────────────────────────────────────
        self.filter_type_var   = tk.StringVar(value="None")
        self.ma_window_var     = tk.IntVar(value=21)
        self.cutoff_low_var    = tk.StringVar(value="100")
        self.cutoff_high_var   = tk.StringVar(value="2000")
        self.filter_order_var  = tk.IntVar(value=4)

        self.window_func_var   = tk.StringVar(value="Hanning")
        self.subtract_dc_var   = tk.BooleanVar(value=True)
        self.n_peaks_var       = tk.IntVar(value=3)
        self.segment_size_var  = tk.StringVar(value="256")

        self.view_mode_var     = tk.StringVar(value="Overview")
        self.max_freq_var      = tk.DoubleVar(value=5000.0)
        self.show_envelope_var = tk.BooleanVar(value=False)
        self.show_peaks_var    = tk.BooleanVar(value=True)

        # ── Build UI ─────────────────────────────────────────
        self._setup_styles()
        self._create_layout()
        self._scan_workspace()
        self._show_placeholder(
            "Select one or more signal files from the navigator\n"
            "to begin response analysis.\n\n"
            "Ctrl+Click to select multiple files for comparison."
        )

    # ═══════════════════════════════════════════════════════════
    #  STYLING
    # ═══════════════════════════════════════════════════════════
    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=DARK_BG, foreground=TEXT_PRIMARY, font=("Segoe UI", 10))
        s.configure("TFrame", background=PANEL_BG)
        s.configure("Panel.TFrame", background=PANEL_BG)
        s.configure("Title.TLabel", background=PANEL_BG, foreground=ACCENT_CYAN,
                     font=("Segoe UI", 11, "bold"))
        s.configure("Field.TLabel", background=PANEL_BG, foreground=TEXT_PRIMARY,
                     font=("Segoe UI", 9))
        s.configure("Muted.TLabel", background=PANEL_BG, foreground=TEXT_MUTED,
                     font=("Segoe UI", 9))
        s.configure("Header.TLabel", background=DARK_BG, foreground=TEXT_PRIMARY,
                     font=("Segoe UI", 13, "bold"))
        s.configure("TCheckbutton", background=PANEL_BG, foreground=TEXT_PRIMARY)
        s.map("TCheckbutton", background=[("active", PANEL_BG)],
              foreground=[("active", ACCENT_CYAN)])
        s.configure("TRadiobutton", background=PANEL_BG, foreground=TEXT_PRIMARY)
        s.map("TRadiobutton", background=[("active", PANEL_BG)],
              foreground=[("active", ACCENT_CYAN)])
        s.configure("Treeview", background="#0e0e18", foreground=TEXT_PRIMARY,
                     fieldbackground="#0e0e18", rowheight=22)
        s.map("Treeview", background=[("selected", ACCENT_CYAN)],
              foreground=[("selected", "#0e0e18")])
        s.configure("Treeview.Heading", background=CARD_BG, foreground=TEXT_PRIMARY,
                     font=("Segoe UI", 9, "bold"))
        s.configure("Accent.TButton", background=ACCENT_CYAN, foreground="#0e0e18",
                     borderwidth=0, font=("Segoe UI", 10, "bold"))
        s.map("Accent.TButton", background=[("active", "#00f0f0")])
        s.configure("Normal.TButton", background=BUTTON_BG, foreground=TEXT_PRIMARY,
                     borderwidth=0)
        s.map("Normal.TButton", background=[("active", "#4a4a7a")])
        s.configure("TCombobox", fieldbackground=CARD_BG, background=CARD_BG,
                     foreground=TEXT_PRIMARY)

    # ═══════════════════════════════════════════════════════════
    #  LAYOUT
    # ═══════════════════════════════════════════════════════════
    def _create_layout(self):
        self.grid_columnconfigure(0, weight=0, minsize=370)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self, style="Panel.TFrame")
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        # ── FILE NAVIGATOR ───────────────────────────────────
        ttk.Label(sidebar, text="📁 File Navigator", style="Title.TLabel"
                  ).pack(anchor="w", padx=12, pady=(10, 4))

        tree_fr = ttk.Frame(sidebar)
        tree_fr.pack(fill=tk.BOTH, expand=True, padx=12, pady=2)
        self.tree = ttk.Treeview(tree_fr, selectmode="extended", show="tree", height=7)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tsb = ttk.Scrollbar(tree_fr, orient="vertical", command=self.tree.yview)
        tsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=tsb.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_file_select)

        ttk.Button(sidebar, text="↻ Rescan Directory", style="Normal.TButton",
                   command=self._scan_workspace).pack(fill=tk.X, padx=12, pady=(2, 4))

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, padx=12, pady=4)

        # ── FILTER SETTINGS ──────────────────────────────────
        ttk.Label(sidebar, text="🔧 Filter Settings", style="Title.TLabel"
                  ).pack(anchor="w", padx=12, pady=(2, 2))

        ft_row = ttk.Frame(sidebar)
        ft_row.pack(fill=tk.X, padx=12)
        ttk.Label(ft_row, text="Type:", style="Field.TLabel").pack(side=tk.LEFT)
        self.filter_combo = ttk.Combobox(ft_row, textvariable=self.filter_type_var,
                                         values=FILTER_TYPES, state="readonly", width=16)
        self.filter_combo.pack(side=tk.LEFT, padx=6)
        self.filter_combo.bind("<<ComboboxSelected>>", self._on_filter_type_change)

        # Dynamic parameter container
        self.filter_params = ttk.Frame(sidebar)
        self.filter_params.pack(fill=tk.X, padx=12, pady=2)

        # -- Moving Average sub-frame --
        self.ma_frame = ttk.Frame(self.filter_params)
        self.lbl_ma_val = ttk.Label(self.ma_frame,
                                    text=f"Window: {self.ma_window_var.get()}", style="Field.TLabel")
        self.lbl_ma_val.pack(anchor="w")
        self.ma_slider = tk.Scale(
            self.ma_frame, from_=1, to=201, orient=tk.HORIZONTAL,
            variable=self.ma_window_var, command=self._on_ma_slider, showvalue=False,
            bg=PANEL_BG, fg=TEXT_PRIMARY, highlightthickness=0,
            activebackground=ACCENT_CYAN, troughcolor="#0e0e18")
        self.ma_slider.pack(fill=tk.X)

        # -- Butterworth sub-frame --
        self.butter_frame = ttk.Frame(self.filter_params)

        cut_row = ttk.Frame(self.butter_frame)
        cut_row.pack(fill=tk.X, pady=2)
        ttk.Label(cut_row, text="Low:", style="Field.TLabel").pack(side=tk.LEFT)
        self.ent_cutoff_low = tk.Entry(cut_row, textvariable=self.cutoff_low_var, width=7,
                                       bg=CARD_BG, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
                                       relief=tk.FLAT)
        self.ent_cutoff_low.pack(side=tk.LEFT, padx=2)
        ttk.Label(cut_row, text="High:", style="Field.TLabel").pack(side=tk.LEFT, padx=(6, 0))
        self.ent_cutoff_high = tk.Entry(cut_row, textvariable=self.cutoff_high_var, width=7,
                                        bg=CARD_BG, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
                                        relief=tk.FLAT)
        self.ent_cutoff_high.pack(side=tk.LEFT, padx=2)
        ttk.Label(cut_row, text="Hz", style="Muted.TLabel").pack(side=tk.LEFT)

        for ent in (self.ent_cutoff_low, self.ent_cutoff_high):
            ent.bind("<Return>", self._trigger_replot_ev)
            ent.bind("<FocusOut>", self._trigger_replot_ev)

        self.lbl_filter_hint = ttk.Label(self.butter_frame, text="", style="Muted.TLabel")
        self.lbl_filter_hint.pack(anchor="w")

        ord_row = ttk.Frame(self.butter_frame)
        ord_row.pack(fill=tk.X, pady=2)
        self.lbl_order_val = ttk.Label(ord_row,
                                       text=f"Order: {self.filter_order_var.get()}", style="Field.TLabel")
        self.lbl_order_val.pack(anchor="w")
        self.order_slider = tk.Scale(
            ord_row, from_=1, to=10, orient=tk.HORIZONTAL,
            variable=self.filter_order_var, command=self._on_order_slider, showvalue=False,
            bg=PANEL_BG, fg=TEXT_PRIMARY, highlightthickness=0,
            activebackground=ACCENT_CYAN, troughcolor="#0e0e18")
        self.order_slider.pack(fill=tk.X)

        self._update_filter_controls()

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, padx=12, pady=4)

        # ── FFT SETTINGS ─────────────────────────────────────
        ttk.Label(sidebar, text="📐 FFT Settings", style="Title.TLabel"
                  ).pack(anchor="w", padx=12, pady=(2, 2))

        fft_r1 = ttk.Frame(sidebar)
        fft_r1.pack(fill=tk.X, padx=12)
        ttk.Label(fft_r1, text="Window:", style="Field.TLabel").pack(side=tk.LEFT)
        wc = ttk.Combobox(fft_r1, textvariable=self.window_func_var, values=WINDOW_FUNCS,
                          state="readonly", width=12)
        wc.pack(side=tk.LEFT, padx=6)
        wc.bind("<<ComboboxSelected>>", self._trigger_replot_ev)

        fft_r2 = ttk.Frame(sidebar)
        fft_r2.pack(fill=tk.X, padx=12, pady=2)
        ttk.Checkbutton(fft_r2, text="Subtract DC Offset",
                        variable=self.subtract_dc_var,
                        command=self._trigger_replot).pack(side=tk.LEFT)

        fft_r3 = ttk.Frame(sidebar)
        fft_r3.pack(fill=tk.X, padx=12, pady=2)
        ttk.Label(fft_r3, text="Peaks:", style="Field.TLabel").pack(side=tk.LEFT)
        tk.Spinbox(fft_r3, from_=0, to=10, width=3, textvariable=self.n_peaks_var,
                   command=self._trigger_replot, bg=CARD_BG, fg=TEXT_PRIMARY,
                   buttonbackground=BUTTON_BG).pack(side=tk.LEFT, padx=4)
        ttk.Label(fft_r3, text="Seg:", style="Field.TLabel").pack(side=tk.LEFT, padx=(8, 0))
        sc = ttk.Combobox(fft_r3, textvariable=self.segment_size_var, values=SEGMENT_SIZES,
                          state="readonly", width=5)
        sc.pack(side=tk.LEFT, padx=4)
        sc.bind("<<ComboboxSelected>>", self._trigger_replot_ev)

        fft_r4 = ttk.Frame(sidebar)
        fft_r4.pack(fill=tk.X, padx=12, pady=2)
        ttk.Checkbutton(fft_r4, text="Show Peak Annotations",
                        variable=self.show_peaks_var,
                        command=self._trigger_replot).pack(side=tk.LEFT)

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, padx=12, pady=4)

        # ── VIEW MODE ────────────────────────────────────────
        ttk.Label(sidebar, text="🔍 View Mode", style="Title.TLabel"
                  ).pack(anchor="w", padx=12, pady=(2, 2))

        vf = ttk.Frame(sidebar)
        vf.pack(fill=tk.X, padx=12)
        for mode in VIEW_MODES:
            ttk.Radiobutton(vf, text=mode, variable=self.view_mode_var, value=mode,
                            command=self._trigger_replot).pack(anchor="w", pady=1)
        ttk.Checkbutton(vf, text="Show Hilbert Envelope",
                        variable=self.show_envelope_var,
                        command=self._trigger_replot).pack(anchor="w", pady=(4, 0))

        freq_fr = ttk.Frame(sidebar)
        freq_fr.pack(fill=tk.X, padx=12, pady=4)
        self.lbl_max_freq = ttk.Label(freq_fr, text="Max Freq: 5000 Hz", style="Field.TLabel")
        self.lbl_max_freq.pack(anchor="w")
        self.freq_slider = tk.Scale(
            freq_fr, from_=50, to=5000, orient=tk.HORIZONTAL,
            variable=self.max_freq_var, command=self._on_freq_slider, showvalue=False,
            bg=PANEL_BG, fg=TEXT_PRIMARY, highlightthickness=0,
            activebackground=ACCENT_CYAN, troughcolor="#0e0e18")
        self.freq_slider.pack(fill=tk.X)

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, padx=12, pady=4)

        # ── STATISTICS PANEL ─────────────────────────────────
        ttk.Label(sidebar, text="📊 Signal Statistics", style="Title.TLabel"
                  ).pack(anchor="w", padx=12, pady=(2, 2))

        # Stats parent container to prevent order shifting on pack/unpack
        self.stats_container = ttk.Frame(sidebar)
        self.stats_container.pack(fill=tk.X, padx=12, pady=2)

        self.primary_selector_frame = ttk.Frame(self.stats_container)
        self.lbl_primary_sel = ttk.Label(self.primary_selector_frame, text="Primary:", style="Field.TLabel")
        self.lbl_primary_sel.pack(side=tk.LEFT)
        self.primary_combo = ttk.Combobox(self.primary_selector_frame, state="readonly", width=22)
        self.primary_combo.pack(side=tk.LEFT, padx=6)
        self.primary_combo.bind("<<ComboboxSelected>>", self._on_primary_change)

        stats_box = tk.Frame(self.stats_container, bg=CARD_BG, padx=10, pady=6)
        stats_box.pack(fill=tk.X, pady=2)
        self.stats_label = tk.Label(stats_box, text="No data loaded.",
                                    font=("Consolas", 9), bg=CARD_BG,
                                    fg=TEXT_MUTED, justify=tk.LEFT, anchor="nw")
        self.stats_label.pack(fill=tk.X)

        # ── EXPORT ───────────────────────────────────────────
        ttk.Button(sidebar, text="💾 Export Processed Data", style="Accent.TButton",
                   command=self._export_data).pack(fill=tk.X, padx=12, pady=(6, 10))

        # ── MAIN PLOT AREA ───────────────────────────────────
        self.plot_frame = ttk.Frame(self, style="Panel.TFrame")
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(1, weight=1)

        self.lbl_plot_title = ttk.Label(self.plot_frame,
                                        text="📈 Response Analysis Dashboard", style="Header.TLabel")
        self.lbl_plot_title.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))

        self.canvas_widget = None
        self.toolbar_widget = None

        import matplotlib.pyplot as plt
        plt.style.use("dark_background")
        self.fig = Figure(figsize=(10, 8), dpi=110, facecolor=DARK_BG)

    # ═══════════════════════════════════════════════════════════
    #  CANVAS MANAGEMENT
    # ═══════════════════════════════════════════════════════════
    def _setup_canvas(self):
        if self.canvas_widget:
            self.canvas_widget.destroy()
        if self.toolbar_widget:
            self.toolbar_widget.destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky="nsew", padx=8, pady=(2, 8))

        tb = ttk.Frame(self.plot_frame, style="Panel.TFrame")
        tb.grid(row=0, column=0, sticky="e", padx=8, pady=(8, 2))
        toolbar = NavigationToolbar2Tk(self.canvas, tb, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.RIGHT)
        self.toolbar_widget = tb

        # Premium interactive crosshairs
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def _show_placeholder(self, text):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(DARK_BG)
        ax.text(0.5, 0.5, text, color=TEXT_MUTED, ha="center", va="center",
                fontsize=12, style="italic")
        ax.axis("off")
        self._setup_canvas()
        self.canvas.draw()

    # ═══════════════════════════════════════════════════════════
    #  FILE MANAGEMENT
    # ═══════════════════════════════════════════════════════════
    def _scan_workspace(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        root_node = self.tree.insert("", "end", text=os.path.basename(self.base_dir), open=True)
        self._populate_tree(root_node, self.base_dir)

    def _populate_tree(self, parent, path):
        try:
            entries = sorted(os.listdir(path),
                             key=lambda x: (not os.path.isdir(os.path.join(path, x)), x.lower()))
        except OSError:
            return
        for name in entries:
            if name.startswith(".") or name in (".venv", "__pycache__", "dist", "build"):
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full):
                if self._has_txt(full):
                    node = self.tree.insert(parent, "end", text=f"📁 {name}", open=False)
                    self._populate_tree(node, full)
            elif name.endswith(".txt"):
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        if "Time" in f.readline():
                            self.tree.insert(parent, "end", text=f"📄 {name}", values=[full])
                except OSError:
                    pass

    @staticmethod
    def _has_txt(path):
        for _, _, files in os.walk(path):
            if any(f.endswith(".txt") for f in files):
                return True
        return False

    def _on_file_select(self, _event):
        paths = []
        for item in self.tree.selection():
            vals = self.tree.item(item, "values")
            if vals:
                paths.append(vals[0])
        if not paths:
            return

        self.signals = []
        for i, p in enumerate(paths):
            sig = self._load_signal(p)
            if sig is not None:
                sig["color"] = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
                self.signals.append(sig)
        if not self.signals:
            self._show_placeholder("Failed to load the selected file(s).")
            return

        self.primary_idx = 0
        nyq = self.signals[0]["nyquist"]
        
        # Configure robust slider limits dynamically
        min_f = min(50.0, nyq / 2.0)
        if min_f >= nyq:
            min_f = nyq / 10.0
        self.freq_slider.config(from_=min_f, to=nyq)
        self.max_freq_var.set(min(5000.0, nyq))
        self.lbl_max_freq.config(text=f"Max Freq: {self.max_freq_var.get():.0f} Hz")

        # Handle multiple files selection with primary dropdown
        if len(self.signals) > 1:
            self.primary_selector_frame.pack(fill=tk.X, pady=2)
            self.primary_combo.config(values=[sig["name"] for sig in self.signals])
            self.primary_combo.current(0)
        else:
            self.primary_selector_frame.pack_forget()

        if len(self.signals) == 1:
            rel = os.path.relpath(self.signals[0]["path"], self.base_dir)
            self.lbl_plot_title.config(text=f"📈 {rel}")
        else:
            self.lbl_plot_title.config(text=f"📈 Comparing {len(self.signals)} signals")

        self._setup_canvas()
        self._trigger_replot()

    def _on_primary_change(self, _event=None):
        idx = self.primary_combo.current()
        if 0 <= idx < len(self.signals):
            self.primary_idx = idx
            nyq = self.signals[idx]["nyquist"]
            
            # Update frequency slider range and values safely
            min_f = min(50.0, nyq / 2.0)
            if min_f >= nyq:
                min_f = nyq / 10.0
            self.freq_slider.config(from_=min_f, to=nyq)
            if self.max_freq_var.get() > nyq:
                self.max_freq_var.set(nyq)
            self.lbl_max_freq.config(text=f"Max Freq: {self.max_freq_var.get():.0f} Hz")
            
            # Recalculate/redraw everything
            self._trigger_replot()

    def _on_mouse_move(self, event):
        if not hasattr(self, "canvas") or not hasattr(self, "crosshairs"):
            return
        
        in_ax = event.inaxes
        changed = False
        
        for ax, v, h, t in self.crosshairs:
            if ax == in_ax and event.xdata is not None and event.ydata is not None:
                x, y = event.xdata, event.ydata
                v.set_xdata([x, x])
                h.set_ydata([y, y])
                
                t.set_text(f"X: {x:.2f}\nY: {y:.2f}")
                t.set_position((x, y))
                
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                ha = "left" if x < (xlim[0] + xlim[1]) / 2.0 else "right"
                va = "bottom" if y < (ylim[0] + ylim[1]) / 2.0 else "top"
                t.set_ha(ha)
                t.set_va(va)
                
                if not v.get_visible():
                    v.set_visible(True)
                    h.set_visible(True)
                    t.set_visible(True)
                    changed = True
            else:
                if v.get_visible():
                    v.set_visible(False)
                    h.set_visible(False)
                    t.set_visible(False)
                    changed = True
                    
        if changed:
            self.canvas.draw_idle()

    def _load_signal(self, path):
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception as e:
            messagebox.showerror("Load Error", f"Cannot read {os.path.basename(path)}:\n{e}")
            return None
        if len(df.columns) < 2:
            return None
        ts = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        amp = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        mask = ts.notna() & amp.notna()
        t = ts[mask].values
        a = amp[mask].values
        if len(t) < 2:
            return None
        N = len(t)
        dt = (t[-1] - t[0]) / (N - 1)
        fs = 1.0 / dt

        orig_freq = orig_spec = None
        if len(df.columns) >= 6:
            of = pd.to_numeric(df.iloc[:, 4], errors="coerce")
            osp = pd.to_numeric(df.iloc[:, 5], errors="coerce")
            m2 = of.notna() & osp.notna()
            if m2.any():
                orig_freq = of[m2].values
                orig_spec = osp[m2].values

        return {
            "path": path, "name": os.path.basename(path),
            "time": t, "raw_amp": a,
            "orig_freq": orig_freq, "orig_spec": orig_spec,
            "fs": fs, "nyquist": fs / 2.0,
            "color": ACCENT_RED,
        }

    # ═══════════════════════════════════════════════════════════
    #  FILTER UI
    # ═══════════════════════════════════════════════════════════
    def _on_filter_type_change(self, _event=None):
        self._update_filter_controls()
        self._trigger_replot()

    def _update_filter_controls(self):
        self.ma_frame.pack_forget()
        self.butter_frame.pack_forget()
        ft = self.filter_type_var.get()
        if ft == "Moving Average":
            self.ma_frame.pack(fill=tk.X, pady=2)
        elif ft in ("Lowpass", "Highpass", "Bandpass"):
            self.butter_frame.pack(fill=tk.X, pady=2)
            hints = {"Lowpass": "Uses High cutoff only",
                     "Highpass": "Uses Low cutoff only",
                     "Bandpass": "Uses both cutoffs"}
            self.lbl_filter_hint.config(text=hints.get(ft, ""))

    def _on_ma_slider(self, val):
        v = int(val)
        if v > 1 and v % 2 == 0:
            v += 1
            self.ma_window_var.set(v)
        self.lbl_ma_val.config(text=f"Window: {v}")
        self._trigger_replot()

    def _on_order_slider(self, _val):
        self.lbl_order_val.config(text=f"Order: {self.filter_order_var.get()}")
        self._trigger_replot()

    # ═══════════════════════════════════════════════════════════
    #  SIGNAL PROCESSING
    # ═══════════════════════════════════════════════════════════
    def _apply_filter(self, raw_amp, fs):
        ft = self.filter_type_var.get()
        if ft == "None":
            return raw_amp.copy()

        if ft == "Moving Average":
            w = self.ma_window_var.get()
            return pd.Series(raw_amp).rolling(window=w, min_periods=1, center=True).mean().values

        # Butterworth
        try:
            order = self.filter_order_var.get()
            nyq = fs / 2.0
            if ft == "Lowpass":
                wn = float(self.cutoff_high_var.get()) / nyq
                if wn <= 0 or wn >= 1:
                    return raw_amp.copy()
                b, a = sp_signal.butter(order, wn, btype="low")
            elif ft == "Highpass":
                wn = float(self.cutoff_low_var.get()) / nyq
                if wn <= 0 or wn >= 1:
                    return raw_amp.copy()
                b, a = sp_signal.butter(order, wn, btype="high")
            elif ft == "Bandpass":
                lo = float(self.cutoff_low_var.get()) / nyq
                hi = float(self.cutoff_high_var.get()) / nyq
                if lo <= 0 or hi >= 1 or lo >= hi:
                    return raw_amp.copy()
                b, a = sp_signal.butter(order, [lo, hi], btype="band")
            else:
                return raw_amp.copy()

            filtered = sp_signal.filtfilt(b, a, raw_amp)
            if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
                return raw_amp.copy()
            return filtered
        except (ValueError, TypeError):
            return raw_amp.copy()

    def _compute_fft(self, time, amplitude):
        N = len(time)
        if N < 2:
            return np.array([]), np.array([])
        dt = (time[-1] - time[0]) / (N - 1)

        sig = amplitude - np.mean(amplitude) if self.subtract_dc_var.get() else amplitude.copy()

        # Window
        wf = self.window_func_var.get()
        if wf == "Hanning":
            win = np.hanning(N)
        elif wf == "Hamming":
            win = np.hamming(N)
        elif wf == "Blackman":
            win = np.blackman(N)
        elif wf == "Flat-top":
            win = sp_signal.windows.flattop(N)
        else:
            win = np.ones(N)

        sig_w = sig * win
        fft_vals = np.fft.rfft(sig_w)
        freqs = np.fft.rfftfreq(N, d=dt)

        win_sum = np.sum(win)
        if win_sum < 1e-15:
            win_sum = N
        fft_amp = 2.0 * np.abs(fft_vals) / win_sum
        fft_amp[0] /= 2.0
        if N % 2 == 0 and len(fft_amp) > 1:
            fft_amp[-1] /= 2.0

        return freqs, fft_amp

    def _compute_psd(self, signal, fs):
        nperseg = min(int(self.segment_size_var.get()), len(signal))
        if nperseg < 4:
            return np.array([]), np.array([])
        try:
            f, Pxx = sp_signal.welch(signal, fs=fs,
                                     window=SCIPY_WIN_MAP.get(self.window_func_var.get(), "hann"),
                                     nperseg=nperseg, noverlap=nperseg // 2)
            return f, Pxx
        except Exception:
            return np.array([]), np.array([])

    def _compute_spectrogram(self, signal, fs):
        nperseg = min(int(self.segment_size_var.get()), len(signal))
        if nperseg < 4:
            return None, None, None
        try:
            f, t, Sxx = sp_signal.spectrogram(
                signal, fs=fs,
                window=SCIPY_WIN_MAP.get(self.window_func_var.get(), "hann"),
                nperseg=nperseg, noverlap=nperseg // 2)
            return f, t, Sxx
        except Exception:
            return None, None, None

    def _compute_envelope(self, signal):
        try:
            return np.abs(sp_signal.hilbert(signal))
        except Exception:
            return None

    def _find_peaks(self, freqs, fft_amp):
        n = self.n_peaks_var.get()
        if n <= 0 or len(fft_amp) < 3:
            return []
        mx = np.max(fft_amp)
        if mx < 1e-15:
            return []
        try:
            indices, _ = sp_signal.find_peaks(
                fft_amp,
                distance=max(3, len(fft_amp) // 100),
                prominence=mx * 0.005)
        except Exception:
            return []
        if len(indices) == 0:
            return []
        top = sorted(indices, key=lambda i: fft_amp[i], reverse=True)[:n]
        return [(idx, freqs[idx], fft_amp[idx]) for idx in top]

    def _estimate_damping(self, freqs, fft_amp, peak_idx):
        if peak_idx <= 0 or peak_idx >= len(fft_amp) - 1:
            return None
        peak_val = fft_amp[peak_idx]
        hp = peak_val / np.sqrt(2)

        # Left crossing
        left = peak_idx - 1
        while left > 0 and fft_amp[left] > hp:
            left -= 1
        if fft_amp[left] > hp:
            return None
        denom_l = fft_amp[left + 1] - fft_amp[left]
        if abs(denom_l) < 1e-20:
            return None
        f1 = freqs[left] + (hp - fft_amp[left]) / denom_l * (freqs[left + 1] - freqs[left])

        # Right crossing
        right = peak_idx + 1
        while right < len(fft_amp) - 1 and fft_amp[right] > hp:
            right += 1
        if fft_amp[right] > hp:
            return None
        denom_r = fft_amp[right - 1] - fft_amp[right]
        if abs(denom_r) < 1e-20:
            return None
        f2 = freqs[right] - (hp - fft_amp[right]) / denom_r * (freqs[right] - freqs[right - 1])

        f0 = freqs[peak_idx]
        if f0 > 0:
            return (f2 - f1) / (2.0 * f0)
        return None

    def _compute_stats(self, raw, filtered, fs, freqs, fft_amp, peaks):
        rms_r = np.sqrt(np.mean(raw ** 2))
        rms_f = np.sqrt(np.mean(filtered ** 2))
        p2p = float(np.max(filtered) - np.min(filtered))
        pk = float(np.max(np.abs(filtered)))
        crest = pk / rms_f if rms_f > 1e-15 else 0.0
        kurt = float(sp_kurtosis(filtered, fisher=True))
        skewness = float(sp_skew(filtered))

        peak_info = []
        for idx, freq, amp in peaks:
            damping = self._estimate_damping(freqs, fft_amp, idx)
            peak_info.append((freq, amp, damping))

        return {
            "rms_raw": rms_r, "rms_filtered": rms_f,
            "peak_to_peak": p2p, "crest_factor": crest,
            "kurtosis": kurt, "skewness": skewness,
            "peaks": peak_info, "fs": fs,
        }

    # ═══════════════════════════════════════════════════════════
    #  EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════
    def _trigger_replot(self):
        if self.signals:
            self._replot()

    def _trigger_replot_ev(self, _event=None):
        self._trigger_replot()

    def _on_freq_slider(self, val):
        self.lbl_max_freq.config(text=f"Max Freq: {float(val):.0f} Hz")
        self._trigger_replot()

    # ═══════════════════════════════════════════════════════════
    #  MASTER REPLOT
    # ═══════════════════════════════════════════════════════════
    def _replot(self):
        processed = []
        for sig in self.signals:
            filtered = self._apply_filter(sig["raw_amp"], sig["fs"])
            freqs, fft_raw = self._compute_fft(sig["time"], sig["raw_amp"])
            _, fft_filt = self._compute_fft(sig["time"], filtered)
            peaks = self._find_peaks(freqs, fft_filt)
            stats = self._compute_stats(sig["raw_amp"], filtered, sig["fs"],
                                        freqs, fft_filt, peaks)
            envelope = self._compute_envelope(filtered) if self.show_envelope_var.get() else None

            processed.append({
                "sig": sig, "filtered": filtered,
                "freqs": freqs, "fft_raw": fft_raw, "fft_filtered": fft_filt,
                "peaks": peaks, "stats": stats, "envelope": envelope,
            })

        self._update_stats(processed[self.primary_idx]["stats"])

        self.fig.clear()
        mode = self.view_mode_var.get()
        if mode == "Overview":
            self._plot_overview(processed)
        elif mode == "Time Domain":
            self._plot_time(processed)
        elif mode == "Frequency Analysis":
            self._plot_freq(processed)
        elif mode == "Spectrogram":
            self._plot_spectro(processed)
        elif mode == "PSD":
            self._plot_psd_view(processed)
        
        self.fig.tight_layout()
        
        # Initialize crosshairs for interactive coordinates tracking
        self.crosshairs = []
        for ax in self.fig.axes:
            if ax.get_label() == "<colorbar>":
                continue
            v = ax.axvline(0, color=TEXT_MUTED, linestyle="--", alpha=0.5, visible=False, linewidth=0.8)
            h = ax.axhline(0, color=TEXT_MUTED, linestyle="--", alpha=0.5, visible=False, linewidth=0.8)
            t = ax.text(0, 0, "", color=TEXT_PRIMARY, fontsize=8,
                        bbox=dict(facecolor=CARD_BG, alpha=0.85, edgecolor=ACCENT_CYAN, boxstyle="round,pad=0.2"),
                        visible=False, zorder=10)
            self.crosshairs.append((ax, v, h, t))
            
        self.canvas.draw()

    # ═══════════════════════════════════════════════════════════
    #  PLOT HELPERS
    # ═══════════════════════════════════════════════════════════
    def _style_ax(self, ax, title, xlabel, ylabel):
        ax.set_facecolor(PLOT_BG)
        ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT_PRIMARY, pad=8)
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT_PRIMARY)
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT_PRIMARY)
        ax.grid(True, linestyle=":", alpha=0.3, color=GRID_CLR)
        ax.tick_params(colors=TEXT_PRIMARY, labelsize=8)

    def _annotate_peaks(self, ax, peaks_info, color):
        for freq, amp, damp in peaks_info:
            ax.axvline(x=freq, color=color, linestyle="--", alpha=0.35, linewidth=0.8)
            ax.scatter(freq, amp, color=color, zorder=5, s=28, edgecolor="white", linewidth=0.5)
            label = f"{freq:.0f} Hz"
            if damp is not None:
                label += f"\nζ={damp:.4f}"
            ax.annotate(label, (freq, amp), textcoords="offset points", xytext=(5, 8),
                        fontsize=7, color=color, fontweight="bold")

    # ═══════════════════════════════════════════════════════════
    #  PLOT VIEWS
    # ═══════════════════════════════════════════════════════════
    def _plot_overview(self, processed):
        """Two subplots: time domain (top) + FFT linear (bottom)."""
        ax1 = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2)
        max_freq = self.max_freq_var.get()
        multi = len(processed) > 1

        for idx, p in enumerate(processed):
            s = p["sig"]
            clr = s["color"]
            lbl = s["name"] if multi else None
            if not multi:
                ax1.plot(s["time"], s["raw_amp"], color="#555570", alpha=0.35,
                         linewidth=0.8, label="Raw")
            ax1.plot(s["time"], p["filtered"], color=clr, linewidth=1.2,
                     label=f"Filtered{' – ' + lbl if lbl else ''}")
            if p["envelope"] is not None and not multi:
                ax1.plot(s["time"], p["envelope"], color=ACCENT_AMBER, linewidth=1.0,
                         alpha=0.7, linestyle="--", label="Envelope")
            ax2.plot(p["freqs"], p["fft_filtered"], color=clr, linewidth=1.2,
                     label=lbl or "Filtered FFT")
            if not multi:
                ax2.plot(p["freqs"], p["fft_raw"], color="#555570", alpha=0.3,
                         linewidth=0.8, label="Raw FFT")
            
            # Draw peak annotations only for the active primary signal (or if single file)
            if self.show_peaks_var.get() and (not multi or idx == self.primary_idx):
                self._annotate_peaks(ax2, p["stats"]["peaks"], clr)

        self._style_ax(ax1, "Time Domain Signal", "Time (s)", "Amplitude")
        ax1.legend(loc="upper right", framealpha=0.2, fontsize=8)
        self._style_ax(ax2, "Frequency Spectrum", "Frequency (Hz)", "Amplitude")
        nyq = self.signals[self.primary_idx]["nyquist"]
        ax2.set_xlim(0, min(max_freq, nyq))
        ax2.legend(loc="upper right", framealpha=0.2, fontsize=8)

    def _plot_time(self, processed):
        """Full-size time domain view with optional envelope."""
        ax = self.fig.add_subplot(1, 1, 1)
        multi = len(processed) > 1

        for p in processed:
            s = p["sig"]
            clr = s["color"]
            lbl = s["name"] if multi else None
            if not multi:
                ax.plot(s["time"], s["raw_amp"], color="#555570", alpha=0.35,
                        linewidth=0.8, label="Raw")
            ax.plot(s["time"], p["filtered"], color=clr, linewidth=1.2,
                    label=f"Filtered{' – ' + lbl if lbl else ''}")
            if p["envelope"] is not None:
                elbl = f"Envelope{' – ' + lbl if lbl else ''}"
                ax.plot(s["time"], p["envelope"], color=ACCENT_AMBER, linewidth=1.0,
                        alpha=0.7, linestyle="--", label=elbl)
                ax.plot(s["time"], -p["envelope"], color=ACCENT_AMBER, linewidth=0.6,
                        alpha=0.4, linestyle="--")

        self._style_ax(ax, "Time Domain Signal", "Time (s)", "Amplitude")
        ax.legend(loc="upper right", framealpha=0.2, fontsize=8)

    def _plot_freq(self, processed):
        """Two subplots: FFT linear (top) + FFT dB (bottom) with peak annotations."""
        ax1 = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2)
        max_freq = self.max_freq_var.get()
        nyq = self.signals[self.primary_idx]["nyquist"]
        multi = len(processed) > 1
        detrend = " (DC removed)" if self.subtract_dc_var.get() else ""

        for idx, p in enumerate(processed):
            s = p["sig"]
            clr = s["color"]
            lbl = s["name"] if multi else None

            # Linear
            if not multi:
                ax1.plot(p["freqs"], p["fft_raw"], color="#555570", alpha=0.3,
                         linewidth=0.8, label="Raw FFT")
                if s["orig_freq"] is not None:
                    ax1.plot(s["orig_freq"], s["orig_spec"], color=ACCENT_BLUE,
                             linestyle="--", alpha=0.6, linewidth=1.0, label="File Spectrum")
            ax1.plot(p["freqs"], p["fft_filtered"], color=clr, linewidth=1.2,
                     label=lbl or "Filtered FFT")
            
            if self.show_peaks_var.get() and (not multi or idx == self.primary_idx):
                self._annotate_peaks(ax1, p["stats"]["peaks"], clr)

            # dB
            fft_db = 20 * np.log10(np.maximum(p["fft_filtered"], 1e-12))
            if not multi:
                db_raw = 20 * np.log10(np.maximum(p["fft_raw"], 1e-12))
                ax2.plot(p["freqs"], db_raw, color="#555570", alpha=0.3,
                         linewidth=0.8, label="Raw FFT (dB)")
            ax2.plot(p["freqs"], fft_db, color=clr, linewidth=1.2,
                     label=lbl or "Filtered FFT (dB)")
            
            # Annotate peaks in dB space only if toggle is on and it is primary
            if self.show_peaks_var.get() and (not multi or idx == self.primary_idx):
                for freq, _, damp in p["stats"]["peaks"]:
                    # Find dB value at peak frequency
                    idx_match = np.argmin(np.abs(p["freqs"] - freq))
                    db_val = fft_db[idx_match]
                    ax2.axvline(x=freq, color=clr, linestyle="--", alpha=0.35, linewidth=0.8)
                    ax2.scatter(freq, db_val, color=clr, zorder=5, s=28,
                                edgecolor="white", linewidth=0.5)
                    plbl = f"{freq:.0f} Hz"
                    if damp is not None:
                        plbl += f"\nζ={damp:.4f}"
                    ax2.annotate(plbl, (freq, db_val), textcoords="offset points",
                                 xytext=(5, 8), fontsize=7, color=clr, fontweight="bold")

        self._style_ax(ax1, f"Frequency Spectrum{detrend}", "Frequency (Hz)", "Amplitude")
        ax1.set_xlim(0, min(max_freq, nyq))
        ax1.legend(loc="upper right", framealpha=0.2, fontsize=8)

        self._style_ax(ax2, f"Frequency Spectrum in dB{detrend}", "Frequency (Hz)",
                       "Amplitude (dB)")
        ax2.set_xlim(0, min(max_freq, nyq))
        ax2.legend(loc="upper right", framealpha=0.2, fontsize=8)

    def _plot_spectro(self, processed):
        """Full-size spectrogram of the primary signal (filtered)."""
        p = processed[self.primary_idx]
        s = p["sig"]
        f, t, Sxx = self._compute_spectrogram(p["filtered"], s["fs"])

        ax = self.fig.add_subplot(1, 1, 1)
        if f is None:
            ax.set_facecolor(PLOT_BG)
            ax.text(0.5, 0.5, "Signal too short for spectrogram\nwith current segment size.",
                    color=TEXT_MUTED, ha="center", va="center", fontsize=12, style="italic")
            ax.axis("off")
            return

        Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-20))
        max_freq = self.max_freq_var.get()
        nyq = s["nyquist"]

        im = ax.pcolormesh(t, f, Sxx_db, cmap="magma", shading="gouraud")
        ax.set_ylim(0, min(max_freq, nyq))
        self._style_ax(ax, f"Spectrogram — {s['name']}", "Time (s)", "Frequency (Hz)")

        cb = self.fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("Power (dB)", fontsize=9, color=TEXT_PRIMARY)
        cb.ax.tick_params(colors=TEXT_PRIMARY, labelsize=8)

    def _plot_psd_view(self, processed):
        """Full-size Welch PSD, multiple signals overlaid."""
        ax = self.fig.add_subplot(1, 1, 1)
        max_freq = self.max_freq_var.get()
        nyq = self.signals[self.primary_idx]["nyquist"]
        multi = len(processed) > 1

        for p in processed:
            s = p["sig"]
            clr = s["color"]
            f, Pxx = self._compute_psd(p["filtered"], s["fs"])
            if len(f) == 0:
                continue
            Pxx_db = 10 * np.log10(np.maximum(Pxx, 1e-20))
            ax.plot(f, Pxx_db, color=clr, linewidth=1.2,
                    label=s["name"] if multi else "Welch PSD")

        self._style_ax(ax, "Power Spectral Density (Welch)", "Frequency (Hz)",
                       "Power/Frequency (dB/Hz)")
        ax.set_xlim(0, min(max_freq, nyq))
        ax.legend(loc="upper right", framealpha=0.2, fontsize=8)

    # ═══════════════════════════════════════════════════════════
    #  STATISTICS DISPLAY
    # ═══════════════════════════════════════════════════════════
    def _update_stats(self, stats):
        lines = [
            f"RMS (Raw):     {stats['rms_raw']:.6f}",
            f"RMS (Filt):    {stats['rms_filtered']:.6f}",
            f"Peak-Peak:     {stats['peak_to_peak']:.6f}",
            f"Crest Factor:  {stats['crest_factor']:.2f}",
            f"Kurtosis:      {stats['kurtosis']:.3f}",
            f"Skewness:      {stats['skewness']:.3f}",
            f"Sampling:      {stats['fs']:.0f} Hz",
            "─" * 28,
        ]
        for i, (freq, _amp, damp) in enumerate(stats["peaks"]):
            d = f"  ζ={damp:.4f}" if damp is not None else ""
            lines.append(f"Peak {i + 1}: {freq:.1f} Hz{d}")
        if not stats["peaks"]:
            lines.append("No peaks detected.")
        self.stats_label.config(text="\n".join(lines), fg=TEXT_PRIMARY)

    # ═══════════════════════════════════════════════════════════
    #  EXPORT
    # ═══════════════════════════════════════════════════════════
    def _export_data(self):
        if not self.signals:
            messagebox.showinfo("Export", "No data loaded to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Processed Data",
        )
        if not path:
            return

        sig = self.signals[self.primary_idx]
        filtered = self._apply_filter(sig["raw_amp"], sig["fs"])
        freqs, fft_raw = self._compute_fft(sig["time"], sig["raw_amp"])
        _, fft_filt = self._compute_fft(sig["time"], filtered)

        time_df = pd.DataFrame({
            "Time_s": sig["time"],
            "Raw_Amplitude": sig["raw_amp"],
            "Filtered_Amplitude": filtered,
        })
        freq_df = pd.DataFrame({
            "Frequency_Hz": freqs,
            "FFT_Raw_Amplitude": fft_raw,
            "FFT_Filtered_Amplitude": fft_filt,
        })
        max_len = max(len(time_df), len(freq_df))
        time_df = time_df.reindex(range(max_len))
        freq_df = freq_df.reindex(range(max_len))
        export_df = pd.concat([time_df, freq_df], axis=1)
        export_df.to_csv(path, index=False)
        messagebox.showinfo("Export", f"Data exported successfully to:\n{path}")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def main():
    app = ResponseAnalyzer()
    app.mainloop()


if __name__ == "__main__":
    main()
