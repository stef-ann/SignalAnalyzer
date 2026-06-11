#!/usr/bin/env python3
"""
Signal Processing Script: Moving Average Filter & FFT Visualization
===================================================================
This script recursively walks through the directory, finds all txt files
containing signal data, applies a moving average filter, computes the FFT
(frequency spectrum) for both raw and filtered signals, and plots the results.

Author: Antigravity AI
Date: 2026-06-11
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Default configuration parameters (can be adjusted)
WINDOW_SIZE = 21          # Window size for the moving average filter
SUBTRACT_MEAN = True     # Subtract mean (DC offset) before computing FFT
SAVE_PLOTS = True         # Save plots next to the original files
SHOW_PLOTS = False        # Interactively display plots (stops execution per plot)
MAX_FREQ_PLOT = 5000      # Max frequency (Hz) to display on the FFT plot (None for full range up to Nyquist)

def moving_average(data, window_size):
    """
    Implements a moving average filter using pandas rolling window.
    This preserves the signal length and avoids edge artifacts using min_periods=1
    and keeps alignment using center=True.
    """
    return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().values

def compute_fft(time, amplitude, subtract_mean=True):
    """
    Computes the single-sided Fast Fourier Transform (FFT) of the signal.
    """
    N = len(time)
    if N < 2:
        return np.array([]), np.array([])
        
    # Calculate sampling interval and frequency
    dt = (time[-1] - time[0]) / (N - 1)
    
    # Detrend by subtracting the mean if specified
    signal = amplitude - np.mean(amplitude) if subtract_mean else amplitude
    
    # Compute real FFT
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=dt)
    
    # Normalized amplitude spectrum
    fft_amp = np.abs(fft_vals) / N
    if len(fft_amp) > 2:
        # Multiply by 2 for single-sided representation (except DC and Nyquist)
        fft_amp[1:-1] *= 2
        
    return freqs, fft_amp

def process_file(file_path, window_size=WINDOW_SIZE, subtract_mean=SUBTRACT_MEAN):
    """
    Reads a single txt data file, processes it, and generates plots.
    """
    print(f"\nProcessing: {file_path}")
    
    # 1. Load data
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"  [Error] Failed to read file: {e}")
        return False

    if len(df.columns) < 2:
        print(f"  [Skip] Less than 2 columns found in file.")
        return False
        
    # Extract columns by index to handle varying column names robustly
    time_col_name = df.columns[0]
    amp_col_name = df.columns[1]
    
    time_series = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    amp_series = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    
    # Clean NaN values
    valid_mask = time_series.notna() & amp_series.notna()
    time = time_series[valid_mask].values
    raw_amp = amp_series[valid_mask].values
    
    if len(time) == 0:
        print(f"  [Skip] No valid numeric signal data found.")
        return False
        
    # Extract original frequency spectrum if columns exist
    orig_freq = None
    orig_spec = None
    if len(df.columns) >= 6:
        orig_freq_series = pd.to_numeric(df.iloc[:, 4], errors='coerce')
        orig_spec_series = pd.to_numeric(df.iloc[:, 5], errors='coerce')
        valid_spec_mask = orig_freq_series.notna() & orig_spec_series.notna()
        if valid_spec_mask.any():
            orig_freq = orig_freq_series[valid_spec_mask].values
            orig_spec = orig_spec_series[valid_spec_mask].values
        
    N = len(time)
    dt = (time[-1] - time[0]) / (N - 1)
    fs = 1.0 / dt
    
    print(f"  - Samples: {N}")
    print(f"  - Sampling interval: {dt*1e6:.2f} us")
    print(f"  - Sampling frequency: {fs:.2f} Hz")
    if orig_freq is not None and len(orig_freq) > 0:
        print(f"  - Original spectrum in file: {len(orig_freq)} bins")
    
    # 2. Apply Moving Average Filter
    filtered_amp = moving_average(raw_amp, window_size)
    
    # 3. Compute FFT
    freqs, fft_raw = compute_fft(time, raw_amp, subtract_mean=subtract_mean)
    _, fft_filtered = compute_fft(time, filtered_amp, subtract_mean=subtract_mean)
    
    if len(freqs) == 0:
        print(f"  [Error] Failed to compute FFT (insufficient data).")
        return False
        
    # Identify dominant frequency
    idx_max_raw = np.argmax(fft_raw)
    idx_max_filtered = np.argmax(fft_filtered)
    dom_freq_raw = freqs[idx_max_raw]
    dom_freq_filtered = freqs[idx_max_filtered]
    
    print(f"  - Raw Dominant Freq: {dom_freq_raw:.2f} Hz (Amp: {fft_raw[idx_max_raw]:.6f})")
    print(f"  - Filtered Dominant Freq: {dom_freq_filtered:.2f} Hz (Amp: {fft_filtered[idx_max_filtered]:.6f})")
    
    # 4. Generate Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), dpi=120)
    fig.patch.set_facecolor('#ffffff')
    
    # Plot formatting styles
    line_raw_color = '#7f8c8d'      # Muted slate gray
    line_filt_color = '#d63031'     # Premium crimson red
    line_raw_fft_color = '#7f8c8d'  # Muted slate gray
    line_filt_fft_color = '#27ae60' # Forest green
    
    # --- Top Plot: Time Domain ---
    ax1.set_facecolor('#fdfdfd')
    ax1.plot(time, raw_amp, label='Raw Signal', color=line_raw_color, alpha=0.5, linewidth=1.0)
    ax1.plot(time, filtered_amp, label=f'Filtered (Moving Avg, w={window_size})', color=line_filt_color, linewidth=1.8)
    
    # Subdir details for title
    rel_path = os.path.relpath(file_path, os.path.dirname(os.path.abspath(__file__)))
    ax1.set_title(f"Time Domain Signal - {rel_path}", fontsize=13, fontweight='bold', pad=12, color='#2c3e50')
    ax1.set_xlabel("Time (seconds)", fontsize=10, labelpad=6)
    ax1.set_ylabel("Amplitude", fontsize=10, labelpad=6)
    ax1.grid(True, linestyle=':', alpha=0.6, color='#bdc3c7')
    ax1.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#dee2e6')
    ax1.tick_params(labelsize=9)
    
    # --- Middle Plot: Frequency Domain (FFT Linear) ---
    ax2.set_facecolor('#fdfdfd')
    if orig_freq is not None and len(orig_freq) > 0:
        ax2.plot(orig_freq, orig_spec, label='Original Raw Spectrum (File)', color='#2980b9', linestyle='--', alpha=0.8, linewidth=1.4)
    ax2.plot(freqs, fft_raw, label='Calculated Raw FFT', color=line_raw_fft_color, alpha=0.5, linewidth=1.0)
    ax2.plot(freqs, fft_filtered, label='Filtered FFT Spectrum', color=line_filt_fft_color, linewidth=1.6)
    
    # Mark peaks on the Linear FFT plot
    if abs(dom_freq_raw - dom_freq_filtered) < 1e-5:
        # Raw and filtered peak are at the same frequency
        ax2.axvline(x=dom_freq_raw, color='#d35400', linestyle='--', alpha=0.6, linewidth=1.2)
        ax2.scatter(dom_freq_raw, fft_raw[idx_max_raw], color='#d35400', zorder=5, s=35, edgecolor='white')
        ax2.scatter(dom_freq_filtered, fft_filtered[idx_max_filtered], color='#27ae60', zorder=6, s=20, edgecolor='white')
        ax2.text(dom_freq_raw, max(fft_raw[idx_max_raw], fft_filtered[idx_max_filtered]) * 1.05, 
                 f"{dom_freq_raw:.1f} Hz", color='#d35400', fontsize=9, fontweight='bold', ha='center')
    else:
        # Separate peaks
        ax2.axvline(x=dom_freq_raw, color=line_raw_fft_color, linestyle='--', alpha=0.5, linewidth=1.0)
        ax2.scatter(dom_freq_raw, fft_raw[idx_max_raw], color='#7f8c8d', zorder=5, s=30, edgecolor='white')
        ax2.text(dom_freq_raw, fft_raw[idx_max_raw] * 1.05, f"{dom_freq_raw:.1f} Hz",
                 color='#555555', fontsize=8, fontweight='bold', ha='center')
                 
        ax2.axvline(x=dom_freq_filtered, color=line_filt_fft_color, linestyle='--', alpha=0.5, linewidth=1.0)
        ax2.scatter(dom_freq_filtered, fft_filtered[idx_max_filtered], color=line_filt_fft_color, zorder=5, s=30, edgecolor='white')
        ax2.text(dom_freq_filtered, fft_filtered[idx_max_filtered] * 1.05, f"{dom_freq_filtered:.1f} Hz",
                 color='#27ae60', fontsize=8, fontweight='bold', ha='center')
    
    # FFT Title details
    detrend_suffix = " (DC Offset Removed)" if subtract_mean else ""
    ax2.set_title(f"Frequency Spectrum Magnitude{detrend_suffix}", fontsize=13, fontweight='bold', pad=12, color='#2c3e50')
    ax2.set_xlabel("Frequency (Hz)", fontsize=10, labelpad=6)
    ax2.set_ylabel("Amplitude / Power Density", fontsize=10, labelpad=6)
    ax2.grid(True, linestyle=':', alpha=0.6, color='#bdc3c7')
    ax2.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#dee2e6')
    ax2.tick_params(labelsize=9)
    
    # Adjust y-limits dynamically to make sure the label fits
    max_peak_val = max(fft_raw[idx_max_raw], fft_filtered[idx_max_filtered])
    if orig_freq is not None and len(orig_freq) > 0:
        max_peak_val = max(max_peak_val, np.max(orig_spec))
    ax2.set_ylim(0, max_peak_val * 1.25)
    
    if MAX_FREQ_PLOT is not None:
        ax2.set_xlim(0, min(MAX_FREQ_PLOT, fs / 2))
        
    # --- Bottom Plot: Frequency Domain (FFT dB Log-Scale) ---
    ax3.set_facecolor('#fdfdfd')
    # Calculate dB magnitude (relative to 1.0)
    db_raw = 20 * np.log10(np.maximum(fft_raw, 1e-12))
    db_filtered = 20 * np.log10(np.maximum(fft_filtered, 1e-12))
    
    if orig_freq is not None and len(orig_freq) > 0:
        # File spectrum is power (amplitude squared), so we use 10 * log10
        db_orig = 10 * np.log10(np.maximum(orig_spec, 1e-24))
        ax3.plot(orig_freq, db_orig, label='Original Raw Spectrum (File, dB)', color='#2980b9', linestyle='--', alpha=0.8, linewidth=1.4)
        
    ax3.plot(freqs, db_raw, label='Calculated Raw FFT (dB)', color=line_raw_fft_color, alpha=0.5, linewidth=1.0)
    ax3.plot(freqs, db_filtered, label='Filtered FFT Spectrum (dB)', color=line_filt_fft_color, linewidth=1.6)
    
    # Mark peaks on the dB FFT plot
    if abs(dom_freq_raw - dom_freq_filtered) < 1e-5:
        # Raw and filtered peak are at the same frequency
        ax3.axvline(x=dom_freq_raw, color='#d35400', linestyle='--', alpha=0.6, linewidth=1.2)
        ax3.scatter(dom_freq_raw, db_raw[idx_max_raw], color='#d35400', zorder=5, s=35, edgecolor='white')
        ax3.scatter(dom_freq_filtered, db_filtered[idx_max_filtered], color='#27ae60', zorder=6, s=20, edgecolor='white')
        ax3.text(dom_freq_raw, max(db_raw[idx_max_raw], db_filtered[idx_max_filtered]) + 5, 
                 f"{dom_freq_raw:.1f} Hz", color='#d35400', fontsize=9, fontweight='bold', ha='center')
    else:
        # Separate peaks
        ax3.axvline(x=dom_freq_raw, color=line_raw_fft_color, linestyle='--', alpha=0.5, linewidth=1.0)
        ax3.scatter(dom_freq_raw, db_raw[idx_max_raw], color='#7f8c8d', zorder=5, s=30, edgecolor='white')
        ax3.text(dom_freq_raw, db_raw[idx_max_raw] + 5, f"{dom_freq_raw:.1f} Hz",
                 color='#555555', fontsize=8, fontweight='bold', ha='center')
                 
        ax3.axvline(x=dom_freq_filtered, color=line_filt_fft_color, linestyle='--', alpha=0.5, linewidth=1.0)
        ax3.scatter(dom_freq_filtered, db_filtered[idx_max_filtered], color=line_filt_fft_color, zorder=5, s=30, edgecolor='white')
        ax3.text(dom_freq_filtered, db_filtered[idx_max_filtered] + 5, f"{dom_freq_filtered:.1f} Hz",
                 color='#27ae60', fontsize=8, fontweight='bold', ha='center')
    
    ax3.set_title(f"Frequency Spectrum Magnitude in dB{detrend_suffix}", fontsize=13, fontweight='bold', pad=12, color='#2c3e50')
    ax3.set_xlabel("Frequency (Hz)", fontsize=10, labelpad=6)
    ax3.set_ylabel("Amplitude (dB ref 1.0)", fontsize=10, labelpad=6)
    ax3.grid(True, linestyle=':', alpha=0.6, color='#bdc3c7')
    ax3.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#dee2e6')
    ax3.tick_params(labelsize=9)
    
    # Adjust y-limits dynamically to make sure the label fits and noise floor doesn't zoom out too far
    max_db_val = max(db_raw[idx_max_raw], db_filtered[idx_max_filtered])
    min_db_val = min(np.min(db_raw), np.min(db_filtered))
    min_db_val = max(min_db_val, -120)  # clamp to -120 dB range
    ax3.set_ylim(min_db_val - 5, max_db_val + 18)
    
    if MAX_FREQ_PLOT is not None:
        ax3.set_xlim(0, min(MAX_FREQ_PLOT, fs / 2))
        
    plt.tight_layout()
    
    # 5. Output Management
    if SAVE_PLOTS:
        plot_path = os.path.splitext(file_path)[0] + "_plot.png"
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"  [Success] Saved visualization to: {plot_path}")
        
    if SHOW_PLOTS:
        plt.show()
        
    plt.close(fig)
    return True

def main():
    print("==========================================================")
    print("Starting Moving Average Filter and FFT Analysis...")
    print(f"Filter Window Size: {WINDOW_SIZE} samples")
    print(f"Subtract DC Offset: {SUBTRACT_MEAN}")
    print("==========================================================")
    
    # Find all .txt files recursively in the workspace
    search_dir = os.path.dirname(os.path.abspath(__file__))
    txt_files = []
    
    for root, dirs, files in os.walk(search_dir):
        # Skip hidden files or folders (e.g. .git)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                
                # Check if it has signal headers to avoid processing config/readme txts
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        header = f.readline()
                        if 'Time' in header:
                            txt_files.append(file_path)
                except Exception:
                    pass
                    
    if not txt_files:
        print("[Warning] No txt files with 'Time' headers found in the directory tree.")
        sys.exit(0)
        
    print(f"Found {len(txt_files)} signal data files to process.")
    
    processed_count = 0
    for file_path in txt_files:
        success = process_file(file_path)
        if success:
            processed_count += 1
            
    print("\n==========================================================")
    print(f"Analysis Complete! Successfully processed {processed_count}/{len(txt_files)} files.")
    print("==========================================================")

if __name__ == '__main__':
    # Allow passing window size via command line argument
    if len(sys.argv) > 1:
        try:
            WINDOW_SIZE = int(sys.argv[1])
            print(f"Overriding default window size. New window size: {WINDOW_SIZE}")
        except ValueError:
            print(f"Invalid window size argument '{sys.argv[1]}'. Using default: {WINDOW_SIZE}")
            
    main()
