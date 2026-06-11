#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Define custom dark theme palette
DARK_BG = "#1e1e24"       # Deep slate gray
SIDEBAR_BG = "#2b2b36"    # Dark panel background
TEXT_COLOR = "#f5f6fa"    # Soft off-white
MUTED_TEXT = "#a4b0be"    # Cool gray
ACCENT_TEAL = "#00b894"   # Vibrant teal green
ACCENT_RED = "#d63031"    # Premium red
ACCENT_BLUE = "#0984e3"   # Bright blue
GRID_COLOR = "#444454"    # Subtle line grid color
BUTTON_BG = "#3d3d5c"     # Slate blue button background

class SignalProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Signal Processing & FFT Real-Time Visualizer")
        self.geometry("1450x950")
        self.configure(bg=DARK_BG)
        
        # Resolve base directory path (supports PyInstaller frozen apps)
        if getattr(sys, 'frozen', False):
            self.base_dir = os.path.dirname(sys.executable)
        else:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # State variables
        self.current_file_path = None
        self.time_data = None
        self.raw_amp_data = None
        self.orig_freq_data = None
        self.orig_spec_data = None
        self.sampling_freq = 0.0
        self.nyquist_freq = 0.0
        
        # Tkinter Control Variables
        self.window_size_var = tk.IntVar(value=21)
        self.subtract_mean_var = tk.BooleanVar(value=True)
        self.show_time_var = tk.BooleanVar(value=True)
        self.show_fft_lin_var = tk.BooleanVar(value=True)
        self.show_fft_db_var = tk.BooleanVar(value=True)
        self.max_freq_var = tk.DoubleVar(value=5000.0)
        self.auto_scale_var = tk.BooleanVar(value=True)
        self.amp_scale_var = tk.DoubleVar(value=1.0)       # Multiplier factor (1.0 = auto / 100%)
        self.fft_lin_scale_var = tk.DoubleVar(value=1.0)
        self.fft_db_min_var = tk.DoubleVar(value=-100.0)
        
        # UI Styling Setup
        self.setup_styles()
        
        # Create Layout structure
        self.create_layout()
        
        # Populate initial file list
        self.scan_workspace()
        
        # Display greeting/placeholder
        self.show_placeholder("Select a text file from the sidebar to visualize signal processing in real-time.")

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure overall widget base styles
        self.style.configure('.', background=DARK_BG, foreground=TEXT_COLOR, font=('Segoe UI', 10))
        self.style.configure('TFrame', background=SIDEBAR_BG)
        
        # Sidebar Panel Labels
        self.style.configure('Sidebar.TFrame', background=SIDEBAR_BG)
        self.style.configure('PanelTitle.TLabel', background=SIDEBAR_BG, foreground=ACCENT_TEAL, font=('Segoe UI', 12, 'bold'))
        self.style.configure('PanelLabel.TLabel', background=SIDEBAR_BG, foreground=TEXT_COLOR, font=('Segoe UI', 10))
        self.style.configure('Muted.TLabel', background=SIDEBAR_BG, foreground=MUTED_TEXT, font=('Segoe UI', 9))
        self.style.configure('Header.TLabel', background=DARK_BG, foreground=TEXT_COLOR, font=('Segoe UI', 14, 'bold'))
        
        # Checkbuttons and Scales
        self.style.configure('TCheckbutton', background=SIDEBAR_BG, foreground=TEXT_COLOR)
        self.style.map('TCheckbutton', background=[('active', SIDEBAR_BG)], foreground=[('active', ACCENT_TEAL)])
        
        # Treeview Styles
        self.style.configure('Treeview', background="#121216", foreground=TEXT_COLOR, fieldbackground="#121216", rowheight=24)
        self.style.map('Treeview', background=[('selected', ACCENT_TEAL)], foreground=[('selected', '#121216')])
        self.style.configure('Treeview.Heading', background="#2d2d30", foreground=TEXT_COLOR, font=('Segoe UI', 10, 'bold'))

        # Buttons
        self.style.configure('Accent.TButton', background=ACCENT_TEAL, foreground="#121216", borderwidth=0, font=('Segoe UI', 10, 'bold'))
        self.style.map('Accent.TButton', background=[('active', '#00d2ad')])
        self.style.configure('Normal.TButton', background=BUTTON_BG, foreground=TEXT_COLOR, borderwidth=0)
        self.style.map('Normal.TButton', background=[('active', '#4e4e75')])

    def create_layout(self):
        # Configure columns: Col 0 for sidebar, Col 1 for main plot area
        self.grid_columnconfigure(0, weight=0, minsize=350)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # ------------------- LEFT SIDEBAR -------------------
        sidebar = ttk.Frame(self, style='Sidebar.TFrame')
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        # File Browser Title
        lbl_file_browser = ttk.Label(sidebar, text="📁 File Navigator", style="PanelTitle.TLabel")
        lbl_file_browser.pack(anchor="w", padx=15, pady=(15, 5))
        
        # Scrollable Treeview for Files
        tree_frame = ttk.Frame(sidebar)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        self.tree = ttk.Treeview(tree_frame, selectmode="browse", show="tree")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.bind("<<TreeviewSelect>>", self.on_file_select)
        
        # Quick Search Directory Refresh
        btn_refresh = ttk.Button(sidebar, text="Scan Workspace Directory", style="Normal.TButton", command=self.scan_workspace)
        btn_refresh.pack(fill=tk.X, padx=15, pady=(2, 10))
        
        # Separator Line
        separator = ttk.Separator(sidebar, orient='horizontal')
        separator.pack(fill=tk.X, padx=15, pady=10)
        
        # ------------------- SIGNAL CONTROLS -------------------
        lbl_controls = ttk.Label(sidebar, text="⚙️ Control Settings", style="PanelTitle.TLabel")
        lbl_controls.pack(anchor="w", padx=15, pady=(5, 5))
        
        # 1. Moving Average Window size
        ma_frame = ttk.Frame(sidebar)
        ma_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.lbl_window_size = ttk.Label(ma_frame, text=f"Moving Average Window: {self.window_size_var.get()}", style="PanelLabel.TLabel")
        self.lbl_window_size.pack(anchor="w")
        
        self.ma_slider = tk.Scale(
            ma_frame, from_=1, to=201, orient=tk.HORIZONTAL, variable=self.window_size_var,
            command=self.on_ma_slider_change, showvalue=False,
            bg=SIDEBAR_BG, fg=TEXT_COLOR, highlightthickness=0,
            activebackground=ACCENT_TEAL, troughcolor="#121216"
        )
        self.ma_slider.pack(fill=tk.X, pady=(2, 5))
        
        # 2. Subtract Mean Checkbox
        self.chk_subtract = ttk.Checkbutton(
            sidebar, text="Subtract DC Mean (Remove Offset)", 
            variable=self.subtract_mean_var, command=self.trigger_replot
        )
        self.chk_subtract.pack(anchor="w", padx=15, pady=5)
        
        # Separator Line
        separator2 = ttk.Separator(sidebar, orient='horizontal')
        separator2.pack(fill=tk.X, padx=15, pady=10)
        
        # ------------------- VIEW & SCALING CONTROLS -------------------
        lbl_view = ttk.Label(sidebar, text="📊 View & Scaling", style="PanelTitle.TLabel")
        lbl_view.pack(anchor="w", padx=15, pady=(5, 5))
        
        # Subplot toggles
        toggles_frame = ttk.Frame(sidebar)
        toggles_frame.pack(fill=tk.X, padx=15, pady=5)
        
        chk_time = ttk.Checkbutton(toggles_frame, text="Time Domain Plot", variable=self.show_time_var, command=self.trigger_replot)
        chk_time.pack(anchor="w", pady=2)
        chk_lin = ttk.Checkbutton(toggles_frame, text="FFT Linear Spectrum", variable=self.show_fft_lin_var, command=self.trigger_replot)
        chk_lin.pack(anchor="w", pady=2)
        chk_db = ttk.Checkbutton(toggles_frame, text="FFT dB Log-Scale", variable=self.show_fft_db_var, command=self.trigger_replot)
        chk_db.pack(anchor="w", pady=2)
        
        # Max Frequency limit slider
        freq_frame = ttk.Frame(sidebar)
        freq_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.lbl_max_freq = ttk.Label(freq_frame, text="Max Plot Frequency: Auto", style="PanelLabel.TLabel")
        self.lbl_max_freq.pack(anchor="w")
        
        self.freq_slider = tk.Scale(
            freq_frame, from_=50, to=5000, orient=tk.HORIZONTAL, variable=self.max_freq_var,
            command=self.on_freq_slider_change, showvalue=False,
            bg=SIDEBAR_BG, fg=TEXT_COLOR, highlightthickness=0,
            activebackground=ACCENT_TEAL, troughcolor="#121216"
        )
        self.freq_slider.pack(fill=tk.X, pady=(2, 5))
        
        # Y Scaling section
        yscale_frame = ttk.Frame(sidebar)
        yscale_frame.pack(fill=tk.X, padx=15, pady=5)
        
        chk_autoscale = ttk.Checkbutton(yscale_frame, text="Auto-Scale Y-Axes", variable=self.auto_scale_var, command=self.on_autoscale_toggle)
        chk_autoscale.pack(anchor="w", pady=(0, 5))
        
        # Manual Scaling Panel (collapsible/hidden when autoscale is on)
        self.manual_scale_frame = ttk.Frame(yscale_frame)
        self.manual_scale_frame.pack(fill=tk.X)
        
        # Time Domain scale multiplier slider
        ttk.Label(self.manual_scale_frame, text="Time Domain Zoom", style="Muted.TLabel").pack(anchor="w")
        self.time_scale_slider = tk.Scale(
            self.manual_scale_frame, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.amp_scale_var,
            command=self.trigger_replot_slider, showvalue=False,
            bg=SIDEBAR_BG, fg=TEXT_COLOR, highlightthickness=0,
            activebackground=ACCENT_TEAL, troughcolor="#121216"
        )
        self.time_scale_slider.pack(fill=tk.X, pady=(0, 5))
        
        # FFT Linear scale multiplier slider
        ttk.Label(self.manual_scale_frame, text="FFT Linear Zoom", style="Muted.TLabel").pack(anchor="w")
        self.fft_lin_scale_slider = tk.Scale(
            self.manual_scale_frame, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.fft_lin_scale_var,
            command=self.trigger_replot_slider, showvalue=False,
            bg=SIDEBAR_BG, fg=TEXT_COLOR, highlightthickness=0,
            activebackground=ACCENT_TEAL, troughcolor="#121216"
        )
        self.fft_lin_scale_slider.pack(fill=tk.X, pady=(0, 5))
        
        # FFT dB minimum scale slider (dynamic range)
        ttk.Label(self.manual_scale_frame, text="FFT dB Floor Limit (dB)", style="Muted.TLabel").pack(anchor="w")
        self.fft_db_min_slider = tk.Scale(
            self.manual_scale_frame, from_=-140.0, to=-20.0, resolution=5, orient=tk.HORIZONTAL, variable=self.fft_db_min_var,
            command=self.trigger_replot_slider, showvalue=False,
            bg=SIDEBAR_BG, fg=TEXT_COLOR, highlightthickness=0,
            activebackground=ACCENT_TEAL, troughcolor="#121216"
        )
        self.fft_db_min_slider.pack(fill=tk.X, pady=(0, 10))
        
        # Hide manual scaling if autoscale is enabled
        self.update_scaling_sliders_visibility()
        
        # Info Panel
        info_frame = ttk.Frame(sidebar)
        info_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=15, pady=15)
        self.lbl_info = ttk.Label(info_frame, text="No file loaded.\nNyquist limit: -- Hz", style="Muted.TLabel", justify=tk.LEFT)
        self.lbl_info.pack(fill=tk.X)
        
        # ------------------- RIGHT PLOT CANVAS AREA -------------------
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=0) # Toolbar row
        self.plot_frame.grid_rowconfigure(1, weight=1) # Plot Canvas row
        
        # Title of visualization area
        self.lbl_plot_title = ttk.Label(self.plot_frame, text="📈 Interactive Visualization Dashboard", style="Header.TLabel", anchor="w")
        self.lbl_plot_title.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        
        # Placeholder for toolbar & canvas
        self.canvas_widget = None
        self.toolbar_widget = None
        
        # Initialize Matplotlib Figure
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(10, 8), dpi=110, facecolor=DARK_BG)

    def setup_canvas(self):
        # Remove old canvas and toolbar if they exist
        if self.canvas_widget:
            self.canvas_widget.destroy()
        if self.toolbar_widget:
            self.toolbar_widget.destroy()
            
        # Create Canvas widget
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        
        # Add Matplotlib Toolbar
        toolbar_container = ttk.Frame(self.plot_frame, style='Sidebar.TFrame')
        toolbar_container.grid(row=0, column=0, sticky="e", padx=10, pady=(10, 5))
        
        # Standard matplotlib NavigationToolbar (we embed it in our custom container)
        # We temporarily disable the label text mapping to avoid Tkinter warnings
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_container, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.toolbar_widget = toolbar_container

    def show_placeholder(self, text):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(DARK_BG)
        ax.text(0.5, 0.5, text, color=MUTED_TEXT, ha='center', va='center', fontsize=12, style='italic')
        ax.axis('off')
        
        self.setup_canvas()
        self.canvas.draw()

    def scan_workspace(self):
        # Clear existing tree elements
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        search_dir = self.base_dir
        root_node = self.tree.insert("", "end", text=f"txt-data ({os.path.basename(search_dir)})", open=True)
        
        # Recursively scan
        self.populate_tree(root_node, search_dir)
        
        # Expand tree view
        self.tree.item(root_node, open=True)

    def populate_tree(self, parent_node, path):
        try:
            items = os.listdir(path)
        except Exception:
            return
            
        # Sort folders first, then files
        items_sorted = sorted(items, key=lambda x: (not os.path.isdir(os.path.join(path, x)), x.lower()))
        
        for item in items_sorted:
            if item.startswith('.') or item == '.venv' or item == '__pycache__':
                continue
                
            abspath = os.path.join(path, item)
            
            if os.path.isdir(abspath):
                # Only insert folder if it contains relevant files recursively
                has_files = self.has_relevant_files(abspath)
                if has_files:
                    folder_node = self.tree.insert(parent_node, 'end', text=f"📁 {item}", open=False)
                    self.populate_tree(folder_node, abspath)
            elif item.endswith('.txt'):
                # Quick verification of headers
                try:
                    with open(abspath, 'r', encoding='utf-8', errors='ignore') as f:
                        header = f.readline()
                        if 'Time' in header:
                            self.tree.insert(parent_node, 'end', text=f"📄 {item}", values=[abspath])
                except Exception:
                    pass

    def has_relevant_files(self, path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.txt'):
                    return True
        return False

    def on_file_select(self, event):
        selected_items = self.tree.selection()
        if not selected_items:
            return
            
        item = selected_items[0]
        values = self.tree.item(item, "values")
        
        if values:
            file_path = values[0]
            self.load_file_data(file_path)

    def load_file_data(self, file_path):
        self.current_file_path = file_path
        
        try:
            df = pd.read_csv(file_path, sep='\t')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            self.show_placeholder(f"Error loading file:\n{file_path}")
            return

        if len(df.columns) < 2:
            messagebox.showwarning("Warning", "The selected file contains less than 2 columns.")
            self.show_placeholder("Invalid file format")
            return
            
        time_series = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        amp_series = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        
        valid_mask = time_series.notna() & amp_series.notna()
        self.time_data = time_series[valid_mask].values
        self.raw_amp_data = amp_series[valid_mask].values
        
        if len(self.time_data) < 2:
            messagebox.showwarning("Warning", "No valid signal data rows found.")
            self.show_placeholder("No valid signal data")
            return
            
        # Parse spectrum columns if present (column 4 and 5)
        self.orig_freq_data = None
        self.orig_spec_data = None
        if len(df.columns) >= 6:
            orig_freq_series = pd.to_numeric(df.iloc[:, 4], errors='coerce')
            orig_spec_series = pd.to_numeric(df.iloc[:, 5], errors='coerce')
            valid_spec_mask = orig_freq_series.notna() & orig_spec_series.notna()
            if valid_spec_mask.any():
                self.orig_freq_data = orig_freq_series[valid_spec_mask].values
                self.orig_spec_data = orig_spec_series[valid_spec_mask].values
                
        # Calculate Sampling Details
        N = len(self.time_data)
        dt = (self.time_data[-1] - self.time_data[0]) / (N - 1)
        self.sampling_freq = 1.0 / dt
        self.nyquist_freq = self.sampling_freq / 2.0
        
        # Dynamically scale Max Frequency slider range
        self.freq_slider.config(to=self.nyquist_freq)
        
        # Update current view variables based on Nyquist limit
        default_max_plot_freq = min(5000.0, self.nyquist_freq)
        self.max_freq_var.set(default_max_plot_freq)
        self.lbl_max_freq.config(text=f"Max Plot Frequency: {default_max_plot_freq:.0f} Hz")
        
        # Reset manual scaling factors to default values
        self.amp_scale_var.set(1.0)
        self.fft_lin_scale_var.set(1.0)
        self.fft_db_min_var.set(-100.0)
        
        # Show file information
        file_name = os.path.basename(file_path)
        info_text = (
            f"Loaded: {file_name}\n"
            f"Samples: {N}\n"
            f"Sampling: {self.sampling_freq:.1f} Hz\n"
            f"Nyquist Limit: {self.nyquist_freq:.1f} Hz"
        )
        self.lbl_info.config(text=info_text)
        
        # Update visualization header
        rel_path = os.path.relpath(file_path, self.base_dir)
        self.lbl_plot_title.config(text=f"📈 {rel_path}")
        
        # Setup plot canvas and render
        self.setup_canvas()
        self.trigger_replot()

    def on_ma_slider_change(self, val):
        val_int = int(val)
        # Enforce odd window sizes for centering properties
        if val_int > 1 and val_int % 2 == 0:
            val_int += 1
            self.window_size_var.set(val_int)
        
        self.lbl_window_size.config(text=f"Moving Average Window: {val_int}")
        self.trigger_replot()

    def on_freq_slider_change(self, val):
        self.lbl_max_freq.config(text=f"Max Plot Frequency: {float(val):.0f} Hz")
        # For frequency limits, we can quickly adjust xlim of axes without doing a full replot!
        # This keeps the dragging experience extremely smooth.
        self.adjust_x_limits()

    def adjust_x_limits(self):
        if not hasattr(self, 'axes_list') or not self.axes_list:
            return
            
        max_freq = self.max_freq_var.get()
        # Find frequency axes (which are indexes 1 and 2 or matching properties)
        for ax, label in self.axes_list:
            if label in ['fft_linear', 'fft_db']:
                ax.set_xlim(0, min(max_freq, self.nyquist_freq))
                
        self.canvas.draw_idle()

    def on_autoscale_toggle(self):
        self.update_scaling_sliders_visibility()
        self.trigger_replot()

    def update_scaling_sliders_visibility(self):
        if self.auto_scale_var.get():
            # Hide manual sliders from geometry manager
            for child in self.manual_scale_frame.winfo_children():
                child.pack_forget()
        else:
            # Re-pack components inside the manual scaling panel
            for child in self.manual_scale_frame.winfo_children():
                child.pack_forget()
                
            # Pack manually
            self.manual_scale_frame.winfo_children()[0].pack(anchor="w") # Label
            self.time_scale_slider.pack(fill=tk.X, pady=(0, 5))
            self.manual_scale_frame.winfo_children()[2].pack(anchor="w") # Label
            self.fft_lin_scale_slider.pack(fill=tk.X, pady=(0, 5))
            self.manual_scale_frame.winfo_children()[4].pack(anchor="w") # Label
            self.fft_db_min_slider.pack(fill=tk.X, pady=(0, 10))

    def trigger_replot_slider(self, val):
        # Helper for sliders where we don't want to choke the thread
        self.trigger_replot()

    def trigger_replot(self):
        if self.time_data is None:
            return
        self.replot_data()

    def compute_moving_average(self, data, window):
        return pd.Series(data).rolling(window=window, min_periods=1, center=True).mean().values

    def compute_fft(self, time, amplitude, subtract_mean):
        N = len(time)
        dt = (time[-1] - time[0]) / (N - 1)
        signal = amplitude - np.mean(amplitude) if subtract_mean else amplitude
        fft_vals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(N, d=dt)
        fft_amp = np.abs(fft_vals) / N
        if len(fft_amp) > 2:
            fft_amp[1:-1] *= 2
        return freqs, fft_amp

    def replot_data(self):
        # Clear the figure structure
        self.fig.clear()
        self.axes_list = []
        
        # Calculate active subplots count
        active_plots = []
        if self.show_time_var.get():
            active_plots.append('time')
        if self.show_fft_lin_var.get():
            active_plots.append('fft_linear')
        if self.show_fft_db_var.get():
            active_plots.append('fft_db')
            
        num_plots = len(active_plots)
        if num_plots == 0:
            self.show_placeholder("Select at least one Plot Type in the sidebar to visualize.")
            return

        # Fetch current UI settings
        window_size = self.window_size_var.get()
        subtract_mean = self.subtract_mean_var.get()
        max_freq_plot = self.max_freq_var.get()
        auto_scale = self.auto_scale_var.get()
        
        # 1. Recalculate signal filter
        filtered_amp = self.compute_moving_average(self.raw_amp_data, window_size)
        
        # 2. Recalculate FFT spectra
        freqs, fft_raw = self.compute_fft(self.time_data, self.raw_amp_data, subtract_mean)
        _, fft_filtered = self.compute_fft(self.time_data, filtered_amp, subtract_mean)
        
        # Peak analysis
        idx_max_raw = np.argmax(fft_raw)
        idx_max_filtered = np.argmax(fft_filtered)
        dom_freq_raw = freqs[idx_max_raw]
        dom_freq_filtered = freqs[idx_max_filtered]
        
        # Start constructing subplots
        plot_idx = 1
        
        # Styling parameters for lines
        raw_color = '#7f8c8d'     # Slate gray
        filt_color = ACCENT_RED   # Premium crimson
        fft_raw_color = '#a4b0be' # Muted gray
        fft_filt_color = ACCENT_TEAL # Teal green
        orig_color = ACCENT_BLUE  # Dark blue spectrum
        
        # --- Time Domain Plot ---
        if 'time' in active_plots:
            ax1 = self.fig.add_subplot(num_plots, 1, plot_idx)
            ax1.set_facecolor("#18181c")
            ax1.plot(self.time_data, self.raw_amp_data, label='Raw Signal', color=raw_color, alpha=0.4, linewidth=1.0)
            ax1.plot(self.time_data, filtered_amp, label=f'Filtered (w={window_size})', color=filt_color, linewidth=1.5)
            
            ax1.set_title("Time Domain Signal", fontsize=11, fontweight='bold', color=TEXT_COLOR, pad=8)
            ax1.set_xlabel("Time (seconds)", fontsize=9, color=TEXT_COLOR)
            ax1.set_ylabel("Amplitude", fontsize=9, color=TEXT_COLOR)
            ax1.grid(True, linestyle=':', alpha=0.3, color=GRID_COLOR)
            ax1.legend(loc='upper right', framealpha=0.2)
            ax1.tick_params(colors=TEXT_COLOR, labelsize=8)
            
            # Y scaling
            if not auto_scale:
                # Zoom in or out relative to maximum peak values
                raw_max = np.max(np.abs(self.raw_amp_data))
                scale_factor = self.amp_scale_var.get()
                ax1.set_ylim(-raw_max * scale_factor, raw_max * scale_factor)
                
            self.axes_list.append((ax1, 'time'))
            plot_idx += 1
            
        # --- FFT Linear plot ---
        if 'fft_linear' in active_plots:
            ax2 = self.fig.add_subplot(num_plots, 1, plot_idx)
            ax2.set_facecolor("#18181c")
            
            # Plot original spectrum from file if present
            if self.orig_freq_data is not None and len(self.orig_freq_data) > 0:
                ax2.plot(self.orig_freq_data, self.orig_spec_data, label='Original Spectrum (File)', color=orig_color, linestyle='--', alpha=0.7, linewidth=1.2)
                
            ax2.plot(freqs, fft_raw, label='Calculated Raw FFT', color=fft_raw_color, alpha=0.4, linewidth=1.0)
            ax2.plot(freqs, fft_filtered, label='Filtered FFT Spectrum', color=fft_filt_color, linewidth=1.5)
            
            # Annotate peak frequency
            max_peak_amp = max(fft_raw[idx_max_raw], fft_filtered[idx_max_filtered])
            if abs(dom_freq_raw - dom_freq_filtered) < 1e-5:
                ax2.axvline(x=dom_freq_raw, color='#d35400', linestyle='--', alpha=0.5, linewidth=1.0)
                ax2.scatter(dom_freq_raw, fft_raw[idx_max_raw], color='#d35400', zorder=5, s=25)
                ax2.text(dom_freq_raw, max_peak_amp * 1.05, f"{dom_freq_raw:.1f} Hz", color='#e67e22', fontsize=8, fontweight='bold', ha='center')
            else:
                ax2.axvline(x=dom_freq_filtered, color=ACCENT_TEAL, linestyle='--', alpha=0.5, linewidth=1.0)
                ax2.scatter(dom_freq_filtered, fft_filtered[idx_max_filtered], color=ACCENT_TEAL, zorder=5, s=25)
                ax2.text(dom_freq_filtered, fft_filtered[idx_max_filtered] * 1.05, f"{dom_freq_filtered:.1f} Hz", color=ACCENT_TEAL, fontsize=8, fontweight='bold', ha='center')

            detrend_suffix = " (DC Removed)" if subtract_mean else ""
            ax2.set_title(f"Frequency Spectrum Magnitude{detrend_suffix}", fontsize=11, fontweight='bold', color=TEXT_COLOR, pad=8)
            ax2.set_xlabel("Frequency (Hz)", fontsize=9, color=TEXT_COLOR)
            ax2.set_ylabel("Amplitude / Density", fontsize=9, color=TEXT_COLOR)
            ax2.grid(True, linestyle=':', alpha=0.3, color=GRID_COLOR)
            ax2.legend(loc='upper right', framealpha=0.2)
            ax2.tick_params(colors=TEXT_COLOR, labelsize=8)
            ax2.set_xlim(0, min(max_freq_plot, self.nyquist_freq))
            
            if auto_scale:
                ax2.set_ylim(0, max_peak_amp * 1.2)
            else:
                scale_factor = self.fft_lin_scale_var.get()
                ax2.set_ylim(0, max_peak_amp * scale_factor)
                
            self.axes_list.append((ax2, 'fft_linear'))
            plot_idx += 1
            
        # --- FFT dB plot ---
        if 'fft_db' in active_plots:
            ax3 = self.fig.add_subplot(num_plots, 1, plot_idx)
            ax3.set_facecolor("#18181c")
            
            # Compute dB relative to 1.0
            db_raw = 20 * np.log10(np.maximum(fft_raw, 1e-12))
            db_filtered = 20 * np.log10(np.maximum(fft_filtered, 1e-12))
            
            if self.orig_freq_data is not None and len(self.orig_freq_data) > 0:
                db_orig = 10 * np.log10(np.maximum(self.orig_spec_data, 1e-24))
                ax3.plot(self.orig_freq_data, db_orig, label='Original Spectrum (File, dB)', color=orig_color, linestyle='--', alpha=0.7, linewidth=1.2)
                
            ax3.plot(freqs, db_raw, label='Calculated Raw FFT (dB)', color=fft_raw_color, alpha=0.4, linewidth=1.0)
            ax3.plot(freqs, db_filtered, label='Filtered FFT Spectrum (dB)', color=fft_filt_color, linewidth=1.5)
            
            # Annotate peak frequency
            max_db = max(db_raw[idx_max_raw], db_filtered[idx_max_filtered])
            if abs(dom_freq_raw - dom_freq_filtered) < 1e-5:
                ax3.axvline(x=dom_freq_raw, color='#d35400', linestyle='--', alpha=0.5, linewidth=1.0)
                ax3.scatter(dom_freq_raw, db_raw[idx_max_raw], color='#d35400', zorder=5, s=25)
                ax3.text(dom_freq_raw, max_db + 4, f"{dom_freq_raw:.1f} Hz", color='#e67e22', fontsize=8, fontweight='bold', ha='center')
            else:
                ax3.axvline(x=dom_freq_filtered, color=ACCENT_TEAL, linestyle='--', alpha=0.5, linewidth=1.0)
                ax3.scatter(dom_freq_filtered, db_filtered[idx_max_filtered], color=ACCENT_TEAL, zorder=5, s=25)
                ax3.text(dom_freq_filtered, db_filtered[idx_max_filtered] + 4, f"{dom_freq_filtered:.1f} Hz", color=ACCENT_TEAL, fontsize=8, fontweight='bold', ha='center')

            detrend_suffix = " (DC Removed)" if subtract_mean else ""
            ax3.set_title(f"Frequency Spectrum in dB{detrend_suffix}", fontsize=11, fontweight='bold', color=TEXT_COLOR, pad=8)
            ax3.set_xlabel("Frequency (Hz)", fontsize=9, color=TEXT_COLOR)
            ax3.set_ylabel("Amplitude (dB ref 1.0)", fontsize=9, color=TEXT_COLOR)
            ax3.grid(True, linestyle=':', alpha=0.3, color=GRID_COLOR)
            ax3.legend(loc='upper right', framealpha=0.2)
            ax3.tick_params(colors=TEXT_COLOR, labelsize=8)
            ax3.set_xlim(0, min(max_freq_plot, self.nyquist_freq))
            
            if auto_scale:
                min_db = min(np.min(db_raw), np.min(db_filtered))
                min_db = max(min_db, -120)  # clamp auto floor to -120 dB
                ax3.set_ylim(min_db - 5, max_db + 15)
            else:
                db_floor = self.fft_db_min_var.get()
                ax3.set_ylim(db_floor, max_db + 15)
                
            self.axes_list.append((ax3, 'fft_db'))
            plot_idx += 1
            
        self.fig.tight_layout()
        self.canvas.draw()

def main():
    app = SignalProcessingApp()
    app.mainloop()

if __name__ == "__main__":
    main()
