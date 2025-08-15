"""
FFT Analysis Module for RoomEQ Audio Processing

This module provides Fast Fourier Transform (FFT) analysis functionality for WAV files,
including spectral analysis, peak detection, frequency band analysis, and windowing functions.
"""

import os
import wave
import numpy as np
from typing import Tuple, Dict, List, Union


def load_wav_file(filepath: str) -> Tuple[np.ndarray, int, Dict]:
    """
    Load WAV file and return audio data, sample rate, and metadata.
    
    Args:
        filepath: Path to the WAV file
        
    Returns:
        Tuple containing:
        - audio_data: Normalized audio data as numpy array [-1, 1]
        - sample_rate: Sample rate in Hz
        - metadata: Dict with file information
        
    Raises:
        RuntimeError: If file cannot be loaded or format is unsupported
    """
    try:
        with wave.open(filepath, 'rb') as wav_file:
            # Get WAV file parameters
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            duration = frames / sample_rate
            
            # Read audio data
            raw_audio = wav_file.readframes(frames)
            
            # Convert to numpy array based on sample width
            if sample_width == 1:
                audio_data = np.frombuffer(raw_audio, dtype=np.uint8)
                # Convert to signed and normalize
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
            elif sample_width == 2:
                audio_data = np.frombuffer(raw_audio, dtype=np.int16)
                # Normalize to [-1, 1]
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio_data = np.frombuffer(raw_audio, dtype=np.int32)
                # Normalize to [-1, 1]
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width} bytes")
            
            # Handle multi-channel audio - convert to mono if needed
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels)
                audio_data = np.mean(audio_data, axis=1)  # Average channels to mono
            
            metadata = {
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': channels,
                'bit_depth': sample_width * 8,
                'total_samples': frames
            }
            
            return audio_data, sample_rate, metadata
            
    except Exception as e:
        raise RuntimeError(f"Failed to load WAV file: {str(e)}")


def compute_fft(audio_data: np.ndarray, sample_rate: int, window_type: str = 'hann', 
                fft_size: int = None, normalize: float = None, points_per_octave: int = None,
                psychoacoustic_smoothing: float = None) -> Dict:
    """
    Compute comprehensive FFT analysis of audio data.
    
    Args:
        audio_data: Normalized audio data [-1, 1]
        sample_rate: Sample rate in Hz
        window_type: Window function ('hann', 'hamming', 'blackman', 'none')
        fft_size: FFT size (power of 2), auto-calculated if None
        normalize: Frequency in Hz to normalize to 0 dB, None for no normalization
        points_per_octave: If specified, summarize FFT into log frequency buckets
        psychoacoustic_smoothing: If specified, apply psychoacoustic smoothing (factor 0.5-2.0)
        
    Returns:
        Dict containing comprehensive FFT analysis results with levels in dB
        
    Raises:
        RuntimeError: If FFT computation fails
    """
    try:
        data_length = len(audio_data)
        
        # Set default FFT size
        if fft_size is None:
            # Use next power of 2 for efficiency, but limit to reasonable size
            fft_size = min(2**int(np.ceil(np.log2(data_length))), 65536)
        
        # Apply window function
        if window_type == 'hann':
            window = np.hanning(min(data_length, fft_size))
        elif window_type == 'hamming':
            window = np.hamming(min(data_length, fft_size))
        elif window_type == 'blackman':
            window = np.blackman(min(data_length, fft_size))
        elif window_type == 'none' or window_type == 'rectangular':
            window = np.ones(min(data_length, fft_size))
        else:
            window = np.hanning(min(data_length, fft_size))  # Default to Hann
        
        # Prepare audio data for FFT
        if data_length >= fft_size:
            # Use first part of audio if longer than FFT size
            windowed_data = audio_data[:fft_size] * window
        else:
            # Zero-pad if audio is shorter than FFT size
            windowed_data = np.zeros(fft_size)
            windowed_data[:data_length] = audio_data * window[:data_length]
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed_data)
        
        # Calculate window normalization factors (for the actual window used)
        actual_window = window[:len(windowed_data)] if len(windowed_data) < len(window) else window
        window_sum = np.sum(actual_window)
        window_power_sum = np.sum(actual_window**2)
        
        # Equivalent Noise Bandwidth (ENBW) correction for the window
        enbw = sample_rate * window_power_sum / (window_sum**2)
        
        # Frequency resolution
        freq_resolution = sample_rate / fft_size
        
        # Compute magnitude spectrum with proper scaling
        magnitude = np.abs(fft_result)
        
        # Scale for single-sided spectrum (except DC and Nyquist)
        if len(magnitude) > 1:
            magnitude[1:] *= 2  # Double all frequencies except DC
            if fft_size % 2 == 0 and len(magnitude) > 1:  # If even FFT size, don't double Nyquist
                magnitude[-1] /= 2
        
        # Normalize by window power sum for proper amplitude scaling
        magnitude = magnitude / np.sqrt(window_power_sum)
        
        # For frequency response analysis, convert magnitude to dB
        # Use 20*log10 for magnitude (not power) to get proper frequency response
        with np.errstate(divide='ignore', invalid='ignore'):
            magnitude_db = 20 * np.log10(magnitude + 1e-20)
            magnitude_db[~np.isfinite(magnitude_db)] = -np.inf
        
        # Compute phase spectrum
        phase = np.angle(fft_result)
        
        # Create frequency axis
        frequencies = np.fft.rfftfreq(fft_size, 1/sample_rate)
        
        # Apply psychoacoustic smoothing if requested
        smoothing_info = {}
        if psychoacoustic_smoothing is not None:
            try:
                magnitude_db = apply_psychoacoustic_smoothing(
                    frequencies, magnitude_db, psychoacoustic_smoothing
                )
                smoothing_info = {
                    "applied": True,
                    "smoothing_factor": float(psychoacoustic_smoothing),
                    "type": "psychoacoustic"
                }
            except Exception as e:
                smoothing_info = {
                    "applied": False,
                    "error": str(e)
                }
        else:
            smoothing_info = {"applied": False}
        
        # Apply normalization if requested
        normalization_info = {}
        if normalize is not None:
            # Find the frequency bin closest to the normalization frequency
            normalize_idx = np.argmin(np.abs(frequencies - normalize))
            normalize_freq_actual = frequencies[normalize_idx]
            normalize_level_db = magnitude_db[normalize_idx]
            
            # Normalize: subtract the reference level from all values
            magnitude_db = magnitude_db - normalize_level_db
            
            normalization_info = {
                "requested_freq": float(normalize),
                "actual_freq": float(normalize_freq_actual),
                "reference_level_db": float(normalize_level_db),
                "applied": True
            }
        else:
            normalization_info = {"applied": False}
        
        # Find peaks
        peak_indices = []
        peak_freqs = []
        peak_magnitudes = []
        
        # Simple peak detection: find local maxima above threshold
        threshold_db = np.max(magnitude_db) - 40  # 40 dB below max
        
        for i in range(1, len(magnitude_db) - 1):
            if (magnitude_db[i] > magnitude_db[i-1] and 
                magnitude_db[i] > magnitude_db[i+1] and 
                magnitude_db[i] > threshold_db):
                peak_indices.append(i)
                peak_freqs.append(frequencies[i])
                peak_magnitudes.append(magnitude_db[i])
        
        # Sort peaks by magnitude (descending)
        peak_data = list(zip(peak_freqs, peak_magnitudes, peak_indices))
        peak_data.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 20 peaks
        top_peaks = peak_data[:20]
        
        # Overall peak frequency and magnitude
        max_idx = np.argmax(magnitude_db)
        peak_frequency = float(frequencies[max_idx])
        peak_magnitude = float(magnitude_db[max_idx])
        
        # Compute spectral statistics
        total_power = np.sum(magnitude**2)
        spectral_centroid = np.sum(frequencies * magnitude**2) / total_power if total_power > 0 else 0
        
        # Frequency bands analysis with named bands
        def analyze_band(f_low: float, f_high: float, name: str) -> Dict:
            band_indices = np.where((frequencies >= f_low) & (frequencies <= f_high))[0]
            if len(band_indices) > 0:
                band_magnitudes = magnitude_db[band_indices]
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
        
        frequency_bands = {
            "sub_bass": analyze_band(20, 60, "Sub-bass"),
            "bass": analyze_band(60, 250, "Bass"),
            "low_midrange": analyze_band(250, 500, "Low midrange"),
            "midrange": analyze_band(500, 2000, "Midrange"),
            "upper_midrange": analyze_band(2000, 4000, "Upper midrange"),
            "presence": analyze_band(4000, 6000, "Presence"),
            "brilliance": analyze_band(6000, 20000, "Brilliance")
        }
        
        # Prepare base result
        result = {
            'fft_size': fft_size,
            'window_type': window_type,
            'sample_rate': sample_rate,
            'frequency_resolution': float(freq_resolution),
            'frequencies': frequencies.tolist(),
            'magnitudes': magnitude_db.tolist(),
            'phases': phase.tolist(),
            'peak_frequency': peak_frequency,
            'peak_magnitude': peak_magnitude,
            'spectral_centroid': float(spectral_centroid),
            'frequency_bands': frequency_bands,
            'normalization': normalization_info,
            'smoothing': smoothing_info,
            'spectral_density': {
                'type': 'Magnitude Spectrum',
                'units': 'dB re FS',
                'description': 'Magnitude spectrum with proper window correction for frequency response analysis',
                'enbw_hz': float(enbw),
                'window_correction_applied': True
            }
        }
        
        # Add logarithmic summarization if requested
        if points_per_octave is not None:
            try:
                log_summary = summarize_fft_log_frequency(
                    frequencies, magnitude_db, points_per_octave
                )
                result['log_frequency_summary'] = log_summary
            except Exception as e:
                # If summarization fails, add error info but continue
                result['log_frequency_summary'] = {
                    'error': f"Summarization failed: {str(e)}",
                    'points_per_octave': points_per_octave
                }
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"FFT computation failed: {str(e)}")


def compute_fft_time_averaged(audio_data: np.ndarray, sample_rate: int, window_type: str = 'hann',
                              fft_size: int = None, overlap: float = 0.5, normalize: float = None,
                              points_per_octave: int = None, psychoacoustic_smoothing: float = None) -> Dict:
    """
    Compute a time-averaged FFT over multiple overlapping windows (Welch-style averaging).

    This method is better suited for wideband, time-varying signals (e.g., sine sweeps),
    providing a stable estimate of overall frequency content across the full duration.

    Args:
        audio_data: Normalized audio data [-1, 1]
        sample_rate: Sample rate in Hz
        window_type: Window function ('hann', 'hamming', 'blackman', 'none')
        fft_size: FFT size (power of 2). If None, choose a reasonable size (<= 65536)
        overlap: Fractional overlap between segments (0.0 - 0.95)
        normalize: Frequency in Hz to normalize to 0 dB, None for no normalization
        points_per_octave: If specified, summarize into log frequency buckets
        psychoacoustic_smoothing: If specified, apply psychoacoustic smoothing (factor 0.5-2.0)

    Returns:
        Dict with averaged FFT results similar to compute_fft(), plus segment metadata
    """
    try:
        x = audio_data.astype(np.float32, copy=False)
        n = len(x)

        if fft_size is None:
            target = min(32768, n)
            fft_size = 1 << int(np.floor(np.log2(target)))
            fft_size = max(1024, min(fft_size, 65536))

        # Window
        win_len = fft_size
        if window_type == 'hann':
            window = np.hanning(win_len)
        elif window_type == 'hamming':
            window = np.hamming(win_len)
        elif window_type == 'blackman':
            window = np.blackman(win_len)
        elif window_type == 'none' or window_type == 'rectangular':
            window = np.ones(win_len)
        else:
            window = np.hanning(win_len)

        # Hop size
        overlap = float(np.clip(overlap, 0.0, 0.95))
        hop = max(1, int(win_len * (1.0 - overlap)))
        if hop <= 0:
            hop = 1

        # If signal is shorter than one window, zero-pad
        if n < win_len:
            padded = np.zeros(win_len, dtype=np.float32)
            padded[:n] = x
            x = padded
            n = win_len

        # Precompute window power and frequency axis
        window_power_sum = float(np.sum(window**2))
        window_sum = float(np.sum(window))
        frequencies = np.fft.rfftfreq(win_len, 1.0 / sample_rate)

        # Accumulate power spectrum across segments
        acc_power = np.zeros(len(frequencies), dtype=np.float64)
        seg_count = 0

        for start in range(0, n - win_len + 1, hop):
            seg = x[start:start + win_len]
            seg_win = seg * window
            fft_result = np.fft.rfft(seg_win)
            mag = np.abs(fft_result)

            # Single-sided scaling
            if len(mag) > 1:
                mag[1:] *= 2
                if win_len % 2 == 0 and len(mag) > 1:
                    mag[-1] /= 2

            # Normalize by window power for amplitude, then power
            mag = mag / np.sqrt(window_power_sum)
            power = mag * mag
            acc_power += power
            seg_count += 1

        if seg_count == 0:
            return compute_fft(audio_data, sample_rate, window_type, fft_size, normalize, 
                             points_per_octave, psychoacoustic_smoothing)

        avg_power = acc_power / seg_count

        # Convert power to dB
        with np.errstate(divide='ignore', invalid='ignore'):
            magnitude_db = 10 * np.log10(avg_power + 1e-20)
            magnitude_db[~np.isfinite(magnitude_db)] = -np.inf

        # Apply psychoacoustic smoothing if requested
        smoothing_info = {}
        if psychoacoustic_smoothing is not None:
            try:
                magnitude_db = apply_psychoacoustic_smoothing(
                    frequencies, magnitude_db, psychoacoustic_smoothing
                )
                smoothing_info = {
                    "applied": True,
                    "smoothing_factor": float(psychoacoustic_smoothing),
                    "type": "psychoacoustic"
                }
            except Exception as e:
                smoothing_info = {
                    "applied": False,
                    "error": str(e)
                }
        else:
            smoothing_info = {"applied": False}

        # Normalization (optional)
        normalization_info = {"applied": False}
        if normalize is not None:
            normalize_idx = int(np.argmin(np.abs(frequencies - normalize)))
            normalize_level_db = float(magnitude_db[normalize_idx])
            magnitude_db = magnitude_db - normalize_level_db
            normalization_info = {
                "requested_freq": float(normalize),
                "actual_freq": float(frequencies[normalize_idx]),
                "reference_level_db": normalize_level_db,
                "applied": True
            }

        # ENBW (informational)
        enbw = sample_rate * window_power_sum / (window_sum**2)

        # Peak stats
        max_idx = int(np.argmax(magnitude_db))
        peak_frequency = float(frequencies[max_idx])
        peak_magnitude = float(magnitude_db[max_idx])

        result = {
            'fft_size': int(win_len),
            'window_type': window_type,
            'sample_rate': int(sample_rate),
            'frequency_resolution': float(sample_rate / win_len),
            'frequencies': frequencies.tolist(),
            'magnitudes': magnitude_db.tolist(),
            'phases': [0.0] * len(frequencies),
            'peak_frequency': peak_frequency,
            'peak_magnitude': peak_magnitude,
            'spectral_centroid': float(np.sum(frequencies * avg_power) / (np.sum(avg_power) + 1e-20)),
            'frequency_bands': {},
            'normalization': normalization_info,
            'smoothing': smoothing_info,
            'spectral_density': {
                'type': 'Averaged Power Spectrum',
                'units': 'dB re FS (power)',
                'description': 'Time-averaged spectrum over overlapping windows',
                'enbw_hz': float(enbw),
                'window_correction_applied': True,
                'n_segments': int(seg_count),
                'overlap': float(overlap)
            }
        }

        if points_per_octave is not None:
            try:
                log_summary = summarize_fft_log_frequency(
                    frequencies, magnitude_db, points_per_octave
                )
                result['log_frequency_summary'] = log_summary
            except Exception as e:
                result['log_frequency_summary'] = {
                    'error': f"Summarization failed: {str(e)}",
                    'points_per_octave': points_per_octave
                }

        return result

    except Exception as e:
        raise RuntimeError(f"Time-averaged FFT computation failed: {str(e)}")


def analyze_wav_file(filepath: str, window_type: str = 'hann', fft_size: int = None,
                     start_time: float = 0.0, duration: float = None, normalize: float = None, 
                     points_per_octave: int = None, psychoacoustic_smoothing: float = None) -> Dict:
    """
    Complete FFT analysis of a WAV file with time windowing support.
    
    Args:
        filepath: Path to WAV file
        window_type: Window function to apply
        fft_size: FFT size (power of 2), auto-calculated if None
        start_time: Start analysis at this time in seconds
        duration: Duration to analyze in seconds, None for entire file
        normalize: Frequency in Hz to normalize to 0 dB, None for no normalization
        points_per_octave: If specified, summarize FFT into log frequency buckets
        psychoacoustic_smoothing: If specified, apply psychoacoustic smoothing (factor 0.5-2.0)
        
    Returns:
        Dict containing file info and FFT analysis
        
    Raises:
        RuntimeError: If analysis fails
        ValueError: If time parameters are invalid
    """
    # Load WAV file
    audio_data, sample_rate, metadata = load_wav_file(filepath)
    
    # Apply time windowing if requested
    if start_time > 0 or duration is not None:
        start_sample = int(start_time * sample_rate)
        
        if start_sample >= len(audio_data):
            raise ValueError(f"start_time {start_time}s exceeds file duration {metadata['duration']:.2f}s")
        
        if duration is not None:
            end_sample = start_sample + int(duration * sample_rate)
            audio_data = audio_data[start_sample:end_sample]
        else:
            audio_data = audio_data[start_sample:]
    
    # Perform FFT analysis
    fft_result = compute_fft(audio_data, sample_rate, window_type, fft_size, normalize, 
                           points_per_octave, psychoacoustic_smoothing)
    
    return {
        'file_info': {
            'filename': os.path.basename(filepath),
            'original_metadata': metadata,
            'analyzed_duration': len(audio_data) / sample_rate,
            'analyzed_samples': len(audio_data),
            'start_time': start_time
        },
        'fft_analysis': fft_result
    }


def get_supported_window_types() -> List[str]:
    """Get list of supported window function types."""
    return ['hann', 'hamming', 'blackman', 'none', 'rectangular']


def validate_fft_parameters(fft_size: int = None, window_type: str = 'hann') -> Dict[str, Union[int, str]]:
    """
    Validate and normalize FFT parameters.
    
    Args:
        fft_size: FFT size to validate
        window_type: Window type to validate
        
    Returns:
        Dict with validated parameters
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Validate window type
    supported_windows = get_supported_window_types()
    if window_type not in supported_windows:
        raise ValueError(f"Unsupported window type '{window_type}'. Supported: {supported_windows}")
    
    # Validate FFT size
    if fft_size is not None:
        if not isinstance(fft_size, int):
            raise ValueError("fft_size must be an integer")
        
        if fft_size < 64 or fft_size > 65536:
            raise ValueError("fft_size must be between 64 and 65536")
        
        # Check if it's a power of 2
        if fft_size & (fft_size - 1) != 0:
            raise ValueError("fft_size must be a power of 2")
    
    return {
        'fft_size': fft_size,
        'window_type': window_type
    }


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
