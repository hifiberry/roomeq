"""
FFT Utilities Module for RoomEQ Audio Processing

This module provides utility functions for FFT post-processing including smoothing,
visualization, and frequency analysis utilities. Graphics functions import matplotlib
only when needed to support systems without matplotlib installed.
"""

import os
import numpy as np
from typing import Tuple, Dict, List, Union
from pathlib import Path


def apply_psychoacoustic_smoothing(frequencies: np.ndarray, magnitudes: np.ndarray,
                                  smoothing_factor: float = 1.0) -> np.ndarray:
    """
    Apply psychoacoustic smoothing to FFT data using frequency-dependent bandwidth.
    
    This smoothing follows critical bands of human hearing, providing narrow bandwidth
    smoothing at low frequencies and wider bandwidth at high frequencies, similar to
    the behavior of the human auditory system.
    
    Args:
        frequencies: Frequency array from FFT
        magnitudes: Magnitude array in dB from FFT
        smoothing_factor: Smoothing strength multiplier (0.5 = less, 2.0 = more)
        
    Returns:
        Smoothed magnitude array in dB
        
    Raises:
        ValueError: If parameters are invalid
    """
    try:
        if len(frequencies) != len(magnitudes):
            raise ValueError("Frequency and magnitude arrays must have same length")
        
        if smoothing_factor <= 0:
            raise ValueError("Smoothing factor must be positive")
        
        # Convert to numpy arrays if needed
        freq = np.asarray(frequencies, dtype=np.float64)
        mag = np.asarray(magnitudes, dtype=np.float64)
        
        # Skip DC component if present
        start_idx = 1 if freq[0] == 0 else 0
        
        smoothed = mag.copy()
        
        # Apply smoothing to each frequency bin
        for i in range(start_idx, len(freq)):
            f_center = freq[i]
            
            if f_center <= 0:
                continue
                
            # Calculate critical band width using Bark scale approximation
            # Critical bandwidth in Hz: CBW ≈ 25 + 75 * (1 + 1.4 * (f/1000)^0.69)^0.64
            # Simplified approximation for computational efficiency
            if f_center < 500:
                # Low frequencies: narrower bandwidth
                cbw = 25 + 75 * (f_center / 1000.0) ** 0.5
            else:
                # Higher frequencies: wider bandwidth following Bark scale
                cbw = 25 + 75 * (1 + 1.4 * (f_center / 1000.0) ** 0.69) ** 0.64
            
            # Apply smoothing factor
            bandwidth = cbw * smoothing_factor
            
            # Define smoothing window around current frequency
            f_low = f_center - bandwidth / 2
            f_high = f_center + bandwidth / 2
            
            # Find frequency bins within the smoothing window
            window_indices = np.where((freq >= f_low) & (freq <= f_high))[0]
            
            if len(window_indices) > 1:
                # Weight function: Gaussian-like weighting centered on f_center
                window_freqs = freq[window_indices]
                weights = np.exp(-0.5 * ((window_freqs - f_center) / (bandwidth / 4)) ** 2)
                
                # Weighted average in linear domain for proper energy conservation
                window_mag_linear = 10 ** (mag[window_indices] / 10.0)
                weighted_avg_linear = np.average(window_mag_linear, weights=weights)
                
                # Convert back to dB
                smoothed[i] = 10 * np.log10(weighted_avg_linear + 1e-20)
        
        return smoothed
        
    except Exception as e:
        raise RuntimeError(f"Psychoacoustic smoothing failed: {str(e)}")


def summarize_fft_log_frequency(frequencies: np.ndarray, magnitudes: np.ndarray, 
                               points_per_octave: int = 16, 
                               f_min: float = 20.0, f_max: float = 20000.0) -> Dict:
    """
    Summarize FFT data into logarithmically spaced frequency buckets using C reference algorithm.
    
    Args:
        frequencies: Frequency array from FFT
        magnitudes: Magnitude array in dB from FFT
        points_per_octave: Number of frequency points per octave
        f_min: Minimum frequency to include (Hz)
        f_max: Maximum frequency to include (Hz)
        
    Returns:
        Dict containing summarized frequency and magnitude data
        
    Raises:
        ValueError: If parameters are invalid
    """
    try:
        # Validate inputs
        if points_per_octave < 1 or points_per_octave > 100:
            raise ValueError("points_per_octave must be between 1 and 100")
        
        if f_min <= 0 or f_max <= f_min:
            raise ValueError("f_min must be positive and f_max must be greater than f_min")
        
        # Clip frequency range to available data
        f_min = max(f_min, frequencies[1])  # Skip DC component
        f_max = min(f_max, frequencies[-1])
        
        if f_min >= f_max:
            raise ValueError("No frequency data available in the specified range")
        
        # Calculate number of octaves and output points
        n_octaves = np.log2(f_max / f_min)
        # Use the C algorithm approach: fixed number of points across the range
        n_points = int(n_octaves * points_per_octave)
        
        # Calculate the frequency spacing factor (from C code)
        x = pow(f_max / f_min, 1.0 / n_points)
        
        # Calculate frequency resolution (video bandwidth)
        freq_resolution = frequencies[1] - frequencies[0]
        
        # Initialize output arrays
        summarized_frequencies = []
        summarized_magnitudes = []
        bin_info = []
        
        # Process each logarithmic bin using C algorithm
        for i in range(n_points):
            # Calculate bin boundaries (from C code)
            fr_start = f_min * pow(x, i)
            fr_end = f_min * pow(x, i + 1)
            
            # Find FFT bins within this range (from C code logic)
            start_bin_idx = int(fr_start / freq_resolution + 0.5)
            
            # Accumulate total power in this logarithmic frequency band
            sum_power = 0.0
            bin_idx = start_bin_idx
            
            # Process all FFT bins that fall within this logarithmic bin range
            while bin_idx < len(frequencies) and frequencies[bin_idx] < fr_end:
                if bin_idx < len(magnitudes):
                    # Convert dB magnitude to power (linear)
                    power_linear = 10**(magnitudes[bin_idx] / 10.0)
                    sum_power += power_linear
                bin_idx += 1
            
            if sum_power > 0:
                # Total power in the logarithmic band
                avg_magnitude_db = 10 * np.log10(sum_power + 1e-20)
            else:
                # No data in this bin
                avg_magnitude_db = -100.0
            
            # Calculate logarithmic average frequency (from C code)
            if fr_end != fr_start and fr_end > 0 and fr_start > 0:
                log_avg_freq = (fr_end - fr_start) / np.log(fr_end / fr_start)
            else:
                log_avg_freq = (fr_start + fr_end) / 2.0
            
            summarized_frequencies.append(float(log_avg_freq))
            summarized_magnitudes.append(float(avg_magnitude_db))
            
            bin_info.append({
                "center_freq": float(log_avg_freq),
                "freq_range": [float(fr_start), float(fr_end)],
                "band_power_linear": float(10**(avg_magnitude_db / 10.0)) if avg_magnitude_db > -1e9 else 0.0,
                "mean_magnitude": float(avg_magnitude_db),
                "min_magnitude": float(avg_magnitude_db),
                "max_magnitude": float(avg_magnitude_db)
            })
        
        return {
            "frequencies": summarized_frequencies,
            "magnitudes": summarized_magnitudes,
            "points_per_octave": points_per_octave,
            "frequency_range": [float(f_min), float(f_max)],
            "n_octaves": float(n_octaves),
            "n_points": len(summarized_frequencies),
            "bin_details": bin_info,
            "algorithm": "C_reference_implementation"
        }
        
    except Exception as e:
        raise RuntimeError(f"FFT summarization failed: {str(e)}")


def create_frequency_plot(frequencies: np.ndarray, magnitudes: np.ndarray, 
                         title: str = "Frequency Response", 
                         output_path: str = None,
                         show_plot: bool = False,
                         freq_range: Tuple[float, float] = (20, 20000),
                         mag_range: Tuple[float, float] = None,
                         grid: bool = True) -> str:
    """
    Create a frequency response plot from FFT data.
    
    Args:
        frequencies: Frequency values in Hz
        magnitudes: Magnitude values in dB
        title: Plot title
        output_path: Path to save the plot (PNG format). If None, uses title + .png
        show_plot: Whether to display the plot
        freq_range: Frequency range to display (min_freq, max_freq) in Hz
        mag_range: Magnitude range to display (min_mag, max_mag) in dB. If None, auto-scale
        grid: Whether to show grid
        
    Returns:
        Path to the saved plot file
        
    Raises:
        RuntimeError: If plotting fails
        ImportError: If matplotlib is not available
    """
    try:
        # Import matplotlib only when needed
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError as e:
        raise ImportError(f"matplotlib is required for plotting but not available: {e}. "
                         f"Install with: pip install matplotlib") from e
    
    try:
        # Create figure with audio-appropriate styling
        plt.figure(figsize=(12, 8))
        
        # Plot frequency response
        plt.semilogx(frequencies, magnitudes, linewidth=1.5, color='#2E86C1', alpha=0.8)
        
        # Set frequency range
        plt.xlim(freq_range[0], freq_range[1])
        
        # Set magnitude range if specified
        if mag_range is not None:
            plt.ylim(mag_range[0], mag_range[1])
        else:
            # Auto-scale with some padding
            mag_min, mag_max = np.min(magnitudes), np.max(magnitudes)
            mag_padding = (mag_max - mag_min) * 0.1
            plt.ylim(mag_min - mag_padding, mag_max + mag_padding)
        
        # Configure axes
        plt.xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        plt.ylabel('Magnitude (dB)', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add grid if requested
        if grid:
            plt.grid(True, which='major', alpha=0.7, linestyle='-', linewidth=0.5)
            plt.grid(True, which='minor', alpha=0.4, linestyle=':', linewidth=0.3)
        
        # Set frequency ticks at standard audio frequencies
        freq_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        freq_ticks = [f for f in freq_ticks if freq_range[0] <= f <= freq_range[1]]
        plt.xticks(freq_ticks, [f'{f}' if f < 1000 else f'{f//1000}k' for f in freq_ticks])
        
        # Improve layout
        plt.tight_layout()
        
        # Determine output path
        if output_path is None:
            output_path = f"{title.replace(' ', '_').replace('/', '_')}.png"
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return output_path
        
    except Exception as e:
        # Make sure to clean up matplotlib resources
        try:
            plt.close()
        except:
            pass
        raise RuntimeError(f"Failed to create frequency plot: {str(e)}")


def plot_wav_frequency_response(wav_path: str, 
                               output_path: str = None,
                               show_plot: bool = False,
                               window_type: str = 'hann',
                               fft_size: int = None,
                               freq_range: Tuple[float, float] = (20, 20000),
                               mag_range: Tuple[float, float] = None,
                               analyzer_module=None) -> str:
    """
    Analyze a WAV file and create a frequency response plot.
    
    Args:
        wav_path: Path to the WAV file
        output_path: Path to save the plot. If None, uses WAV filename + .png
        show_plot: Whether to display the plot
        window_type: FFT window type ('hann', 'hamming', 'blackman', etc.)
        fft_size: FFT size. If None, uses default
        freq_range: Frequency range to display (min_freq, max_freq) in Hz
        mag_range: Magnitude range to display (min_mag, max_mag) in dB
        analyzer_module: FFT analyzer module to use (defaults to fft.py)
        
    Returns:
        Path to the saved plot file
        
    Raises:
        RuntimeError: If analysis or plotting fails
        ImportError: If matplotlib is not available
    """
    try:
        # Use the provided analyzer module or default to fft
        if analyzer_module is None:
            from . import fft as analyzer_module
        
        # Analyze WAV file
        result = analyzer_module.analyze_wav_file(wav_path, window_type=window_type, fft_size=fft_size)
        
        # Extract frequency and magnitude data from nested structure
        fft_data = result['fft_analysis']
        frequencies = np.array(fft_data['frequencies'])
        magnitudes = np.array(fft_data['magnitudes'])
        
        # Filter to frequency range
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        frequencies = frequencies[freq_mask]
        magnitudes = magnitudes[freq_mask]
        
        # Generate title from filename
        wav_filename = os.path.splitext(os.path.basename(wav_path))[0]
        title = f"Frequency Response: {wav_filename}"
        
        # Generate output path if not provided
        if output_path is None:
            output_dir = os.path.dirname(wav_path)
            output_path = os.path.join(output_dir, f"{wav_filename}.png")
        
        # Create and save plot
        plot_path = create_frequency_plot(
            frequencies, magnitudes, 
            title=title,
            output_path=output_path,
            show_plot=show_plot,
            freq_range=freq_range,
            mag_range=mag_range
        )
        
        return plot_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to plot WAV frequency response: {str(e)}")


def analyze_frequency_bands(frequencies: np.ndarray, magnitudes: np.ndarray) -> Dict:
    """
    Analyze standard audio frequency bands and return statistics.
    
    Args:
        frequencies: Frequency array from FFT
        magnitudes: Magnitude array in dB from FFT
        
    Returns:
        Dict with frequency band analysis
    """
    def analyze_band(f_low: float, f_high: float, name: str) -> Dict:
        band_indices = np.where((frequencies >= f_low) & (frequencies <= f_high))[0]
        if len(band_indices) > 0:
            band_magnitudes = magnitudes[band_indices]
            band_frequencies = frequencies[band_indices]
            avg_magnitude = float(np.mean(band_magnitudes))
            peak_idx = np.argmax(band_magnitudes)
            peak_freq = float(band_frequencies[peak_idx])
            return {
                "range": f"{f_low}-{f_high} Hz",
                "avg_magnitude": avg_magnitude,
                "peak_frequency": peak_freq
            }
        else:
            return {
                "range": f"{f_low}-{f_high} Hz",
                "avg_magnitude": -100.0,
                "peak_frequency": f_low
            }
    
    return {
        "sub_bass": analyze_band(20, 60, "Sub-bass"),
        "bass": analyze_band(60, 250, "Bass"),
        "low_midrange": analyze_band(250, 500, "Low midrange"),
        "midrange": analyze_band(500, 2000, "Midrange"),
        "upper_midrange": analyze_band(2000, 4000, "Upper midrange"),
        "presence": analyze_band(4000, 6000, "Presence"),
        "brilliance": analyze_band(6000, 20000, "Brilliance")
    }


def find_peaks(frequencies: np.ndarray, magnitudes: np.ndarray, 
               threshold_db: float = None, min_peak_distance: float = 50.0) -> Dict:
    """
    Find prominent peaks in the frequency spectrum.
    
    Args:
        frequencies: Frequency array from FFT
        magnitudes: Magnitude array in dB from FFT  
        threshold_db: Minimum peak height in dB below max. If None, uses -40dB
        min_peak_distance: Minimum distance between peaks in Hz
        
    Returns:
        Dict with peak information
    """
    if threshold_db is None:
        threshold_db = np.max(magnitudes) - 40  # 40 dB below max
    
    peak_indices = []
    peak_freqs = []
    peak_magnitudes = []
    
    # Simple peak detection: find local maxima above threshold
    for i in range(1, len(magnitudes) - 1):
        if (magnitudes[i] > magnitudes[i-1] and 
            magnitudes[i] > magnitudes[i+1] and 
            magnitudes[i] > threshold_db):
            
            # Check minimum distance from existing peaks
            too_close = False
            for existing_freq in peak_freqs:
                if abs(frequencies[i] - existing_freq) < min_peak_distance:
                    too_close = True
                    break
            
            if not too_close:
                peak_indices.append(i)
                peak_freqs.append(frequencies[i])
                peak_magnitudes.append(magnitudes[i])
    
    # Sort peaks by magnitude (descending)
    peak_data = list(zip(peak_freqs, peak_magnitudes, peak_indices))
    peak_data.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 20 peaks
    top_peaks = peak_data[:20]
    
    return {
        "peaks": [{"frequency": float(f), "magnitude": float(m), "index": int(idx)} 
                  for f, m, idx in top_peaks],
        "threshold_db": float(threshold_db),
        "total_peaks_found": len(peak_data)
    }


def fft_diff(fft_result1: Dict, fft_result2: Dict, 
             title: str = "FFT Difference",
             description: str = None) -> Dict:
    """
    Create a difference between two FFT analysis results.
    
    This function computes the difference between two FFT results, typically used
    to compare input vs output signals, or measurement vs reference signals.
    The difference is computed as: fft_result1 - fft_result2 (in dB domain).
    
    Args:
        fft_result1: First FFT analysis result (e.g., recording/output signal)
        fft_result2: Second FFT analysis result (e.g., signal/input signal)
        title: Title for the difference result
        description: Optional description of what the difference represents
        
    Returns:
        Dict containing the difference analysis with same structure as input FFT results
        
    Raises:
        ValueError: If FFT results are incompatible or invalid
        RuntimeError: If difference computation fails
    """
    try:
        # Validate input FFT results
        if not isinstance(fft_result1, dict) or not isinstance(fft_result2, dict):
            raise ValueError("Both inputs must be FFT analysis dictionaries")
        
        # Extract frequency and magnitude data
        # Extract frequency and magnitude data from FFT results
        # First, try to get data from fft_analysis section
        if ('fft_analysis' in fft_result1 and 'frequencies' in fft_result1['fft_analysis'] 
            and 'magnitudes' in fft_result1['fft_analysis']):
            # Use full resolution data from fft_analysis
            freqs1 = np.array(fft_result1['fft_analysis']['frequencies'])
            mags1 = np.array(fft_result1['fft_analysis']['magnitudes'])
        elif ('fft_analysis' in fft_result1 and 'log_frequency_summary' in fft_result1['fft_analysis']):
            # Use log frequency summary from fft_analysis
            log_data1 = fft_result1['fft_analysis']['log_frequency_summary']
            freqs1 = np.array(log_data1['frequencies'])
            mags1 = np.array(log_data1['magnitudes'])
        elif 'frequencies' in fft_result1 and 'magnitudes' in fft_result1:
            # Legacy: try root level (backwards compatibility)
            freqs1 = np.array(fft_result1['frequencies'])
            mags1 = np.array(fft_result1['magnitudes'])
        elif 'log_frequency_summary' in fft_result1:
            # Legacy: try root level log frequency summary
            log_data1 = fft_result1['log_frequency_summary']
            freqs1 = np.array(log_data1['frequencies'])
            mags1 = np.array(log_data1['magnitudes'])
        else:
            raise ValueError("First FFT result missing frequency/magnitude data")
        
        if ('fft_analysis' in fft_result2 and 'frequencies' in fft_result2['fft_analysis'] 
            and 'magnitudes' in fft_result2['fft_analysis']):
            # Use full resolution data from fft_analysis
            freqs2 = np.array(fft_result2['fft_analysis']['frequencies'])
            mags2 = np.array(fft_result2['fft_analysis']['magnitudes'])
        elif ('fft_analysis' in fft_result2 and 'log_frequency_summary' in fft_result2['fft_analysis']):
            # Use log frequency summary from fft_analysis
            log_data2 = fft_result2['fft_analysis']['log_frequency_summary']
            freqs2 = np.array(log_data2['frequencies'])
            mags2 = np.array(log_data2['magnitudes'])
        elif 'frequencies' in fft_result2 and 'magnitudes' in fft_result2:
            # Legacy: try root level (backwards compatibility)
            freqs2 = np.array(fft_result2['frequencies'])
            mags2 = np.array(fft_result2['magnitudes'])
        elif 'log_frequency_summary' in fft_result2:
            # Legacy: try root level log frequency summary
            log_data2 = fft_result2['log_frequency_summary']
            freqs2 = np.array(log_data2['frequencies'])
            mags2 = np.array(log_data2['magnitudes'])
        else:
            raise ValueError("Second FFT result missing frequency/magnitude data")
        
        # Determine common frequency range
        freq_min = max(freqs1.min(), freqs2.min())
        freq_max = min(freqs1.max(), freqs2.max())
        
        if freq_min >= freq_max:
            raise ValueError("No overlapping frequency range between FFT results")
        
        # Use the frequency grid from the first result, but clip to common range
        freq_mask1 = (freqs1 >= freq_min) & (freqs1 <= freq_max)
        target_freqs = freqs1[freq_mask1]
        target_mags1 = mags1[freq_mask1]
        
        # Interpolate second result to match first result's frequency grid
        target_mags2 = np.interp(target_freqs, freqs2, mags2)
        
        # Compute difference: result1 - result2 (in dB domain)
        diff_mags = target_mags1 - target_mags2
        
        # Calculate statistics of the difference
        rms_diff = np.sqrt(np.mean(diff_mags**2))
        max_diff = np.max(np.abs(diff_mags))
        mean_diff = np.mean(diff_mags)
        
        # Find peaks in the difference
        diff_peaks = find_peaks(target_freqs, np.abs(diff_mags))
        
        # Analyze frequency bands for the difference
        diff_bands = analyze_frequency_bands(target_freqs, diff_mags)
        
        # Find overall peak difference
        max_idx = np.argmax(np.abs(diff_mags))
        peak_frequency = float(target_freqs[max_idx])
        peak_magnitude = float(diff_mags[max_idx])
        
        # Create result structure similar to original FFT results
        diff_result = {
            'title': title,
            'description': description or f"Difference analysis: {title}",
            'diff_type': 'magnitude_difference_db',
            'sample_rate': fft_result1.get('sample_rate', fft_result2.get('sample_rate', 44100)),
            'frequencies': target_freqs.tolist(),
            'magnitudes': diff_mags.tolist(),
            'phases': [0.0] * len(target_freqs),  # Phase difference not computed
            'peak_frequency': peak_frequency,
            'peak_magnitude': peak_magnitude,
            'frequency_bands': diff_bands,
            'peaks': diff_peaks,
            'statistics': {
                'rms_difference_db': float(rms_diff),
                'max_difference_db': float(max_diff),
                'mean_difference_db': float(mean_diff),
                'frequency_range': [float(freq_min), float(freq_max)],
                'n_points': len(target_freqs)
            },
            'source_info': {
                'result1_title': fft_result1.get('title', 'Result 1'),
                'result2_title': fft_result2.get('title', 'Result 2'),
                'result1_peak_freq': fft_result1.get('peak_frequency', 0.0),
                'result2_peak_freq': fft_result2.get('peak_frequency', 0.0),
            },
            'spectral_density': {
                'type': 'FFT Magnitude Difference',
                'units': 'dB difference',
                'description': 'Magnitude difference between two FFT results',
                'computation': 'result1_db - result2_db'
            }
        }
        
        return diff_result
        
    except Exception as e:
        raise RuntimeError(f"FFT difference computation failed: {str(e)}")


def plot_fft_diff(diff_result: Dict,
                  output_path: str = None,
                  show_plot: bool = False,
                  freq_range: Tuple[float, float] = (20, 20000),
                  mag_range: Tuple[float, float] = None) -> str:
    """
    Create a plot of FFT difference analysis.
    
    Args:
        diff_result: Result from fft_diff() function
        output_path: Path to save the plot. If None, uses diff title + .png
        show_plot: Whether to display the plot
        freq_range: Frequency range to display (min_freq, max_freq) in Hz
        mag_range: Magnitude range to display (min_mag, max_mag) in dB
        
    Returns:
        Path to the saved plot file
        
    Raises:
        RuntimeError: If plotting fails
        ImportError: If matplotlib is not available
    """
    try:
        # Import matplotlib only when needed
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError as e:
        raise ImportError(f"matplotlib is required for plotting but not available: {e}. "
                         f"Install with: pip install matplotlib") from e
    
    try:
        # Extract data
        frequencies = np.array(diff_result['frequencies'])
        magnitudes = np.array(diff_result['magnitudes'])
        title = diff_result.get('title', 'FFT Difference')
        
        # Filter to frequency range
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        frequencies = frequencies[freq_mask]
        magnitudes = magnitudes[freq_mask]
        
        # Create figure with difference-appropriate styling
        plt.figure(figsize=(12, 8))
        
        # Plot difference with color coding (positive = red, negative = blue)
        colors = ['red' if m > 0 else 'blue' for m in magnitudes]
        plt.semilogx(frequencies, magnitudes, linewidth=1.5, color='purple', alpha=0.8)
        
        # Add zero reference line
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        # Set frequency range
        plt.xlim(freq_range[0], freq_range[1])
        
        # Set magnitude range if specified
        if mag_range is not None:
            plt.ylim(mag_range[0], mag_range[1])
        else:
            # Auto-scale with some padding
            mag_min, mag_max = np.min(magnitudes), np.max(magnitudes)
            mag_padding = max(abs(mag_max - mag_min) * 0.1, 1.0)  # At least 1dB padding
            plt.ylim(mag_min - mag_padding, mag_max + mag_padding)
        
        # Configure axes
        plt.xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        plt.ylabel('Magnitude Difference (dB)', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        plt.grid(True, which='major', alpha=0.7, linestyle='-', linewidth=0.5)
        plt.grid(True, which='minor', alpha=0.4, linestyle=':', linewidth=0.3)
        
        # Set frequency ticks at standard audio frequencies
        freq_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        freq_ticks = [f for f in freq_ticks if freq_range[0] <= f <= freq_range[1]]
        plt.xticks(freq_ticks, [f'{f}' if f < 1000 else f'{f//1000}k' for f in freq_ticks])
        
        # Add statistics text box
        if 'statistics' in diff_result:
            stats = diff_result['statistics']
            stats_text = (f"RMS Diff: {stats['rms_difference_db']:.2f} dB\n"
                          f"Max Diff: {stats['max_difference_db']:.2f} dB\n"
                          f"Mean Diff: {stats['mean_difference_db']:.2f} dB")
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Improve layout
        plt.tight_layout()
        
        # Determine output path
        if output_path is None:
            safe_title = title.replace(' ', '_').replace('/', '_')
            output_path = f"{safe_title}.png"
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return output_path
        
    except Exception as e:
        # Make sure to clean up matplotlib resources
        try:
            plt.close()
        except:
            pass
        raise RuntimeError(f"Failed to create FFT difference plot: {str(e)}")
