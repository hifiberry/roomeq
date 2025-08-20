"""
FFT Analysis Module for RoomEQ Audio Processing

This module provides Fast Fourier Transform (FFT) analysis functionality for WAV files,
including spectral analysis, peak detection, frequency band analysis, and windowing functions.
Core FFT computation and WAV file reading - smoothing and plotting moved to fft_utils.py.
Uses SciPy FFT for improved performance and accuracy.
"""

import os
import wave
import numpy as np
from typing import Tuple, Dict, List, Union
from pathlib import Path
from . import fft_utils

# Use SciPy FFT for better performance and accuracy
try:
    from scipy.fft import fft, rfft, rfftfreq
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    # Fallback to numpy FFT if SciPy is not available
    from numpy.fft import fft, rfft, rfftfreq
    from numpy import signal
    HAS_SCIPY = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


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
    # First try with standard wave module
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
            
    except Exception as wave_error:
        # If standard wave module fails, try soundfile as fallback
        if HAS_SOUNDFILE and ("unknown format" in str(wave_error) or "format" in str(wave_error).lower()):
            try:
                # Use soundfile to load the WAV file
                audio_data, sample_rate = sf.read(filepath, dtype='float32')
                
                # Handle multi-channel audio - convert to mono if needed
                if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                    channels = audio_data.shape[1]
                    audio_data = np.mean(audio_data, axis=1)  # Average channels to mono
                else:
                    channels = 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
                
                # Calculate metadata
                duration = len(audio_data) / sample_rate
                total_samples = len(audio_data)
                
                # Try to get bit depth from file info
                try:
                    info = sf.info(filepath)
                    bit_depth = getattr(info, 'subtype_info', '').split('bit')[0] or '32'  # Default for float
                    try:
                        bit_depth = int(bit_depth)
                    except (ValueError, TypeError):
                        bit_depth = 32  # Default for float32
                except:
                    bit_depth = 32  # Default fallback
                
                metadata = {
                    'duration': duration,
                    'sample_rate': int(sample_rate),
                    'channels': channels,
                    'bit_depth': bit_depth,
                    'total_samples': total_samples,
                    'loaded_with': 'soundfile'
                }
                
                return audio_data, int(sample_rate), metadata
                
            except Exception as sf_error:
                raise RuntimeError(f"Failed to load WAV file with both wave and soundfile modules. "
                                 f"Wave error: {wave_error}. Soundfile error: {sf_error}")
        else:
            if not HAS_SOUNDFILE:
                raise RuntimeError(f"Failed to load WAV file: {wave_error}. "
                                 f"For better format support, install soundfile: pip install soundfile")
            else:
                raise RuntimeError(f"Failed to load WAV file: {wave_error}")
    
    except Exception as e:
        raise RuntimeError(f"Failed to load WAV file: {str(e)}")


def compute_fft(audio_data: np.ndarray, sample_rate: int, window_type: str = 'hann', 
                fft_size: int = None, normalize: float = None, points_per_octave: int = None) -> Dict:
    """
    Compute comprehensive FFT analysis of audio data.
    
    Args:
        audio_data: Normalized audio data [-1, 1]
        sample_rate: Sample rate in Hz
        window_type: Window function ('hann', 'hamming', 'blackman', 'none')
        fft_size: FFT size (power of 2), auto-calculated if None
        normalize: Frequency in Hz to normalize to 0 dB, None for no normalization
        points_per_octave: If specified, summarize FFT into log frequency buckets
        
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
        
        # Apply window function using SciPy signal module
        window_length = min(data_length, fft_size)
        if window_type == 'hann':
            if HAS_SCIPY:
                window = signal.windows.hann(window_length)
            else:
                window = np.hanning(window_length)
        elif window_type == 'hamming':
            if HAS_SCIPY:
                window = signal.windows.hamming(window_length)
            else:
                window = np.hamming(window_length)
        elif window_type == 'blackman':
            if HAS_SCIPY:
                window = signal.windows.blackman(window_length)
            else:
                window = np.blackman(window_length)
        elif window_type == 'none' or window_type == 'rectangular':
            window = np.ones(window_length)
        else:
            # Default to Hann window
            if HAS_SCIPY:
                window = signal.windows.hann(window_length)
            else:
                window = np.hanning(window_length)
        
        # Prepare audio data for FFT
        if data_length >= fft_size:
            # Use first part of audio if longer than FFT size
            windowed_data = audio_data[:fft_size] * window
        else:
            # Zero-pad if audio is shorter than FFT size
            windowed_data = np.zeros(fft_size)
            windowed_data[:data_length] = audio_data * window[:data_length]
        
        # Compute FFT using SciPy (or numpy fallback)
        fft_result = rfft(windowed_data)
        
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
        
        # Normalize by window sum for proper amplitude scaling (coherent gain correction)
        magnitude = magnitude / window_sum
        
        # For frequency response analysis, convert magnitude to dB
        # Use 20*log10 for magnitude (not power) to get proper frequency response
        with np.errstate(divide='ignore', invalid='ignore'):
            magnitude_db = 20 * np.log10(magnitude + 1e-20)
            magnitude_db[~np.isfinite(magnitude_db)] = -np.inf
        
        # Compute phase spectrum
        phase = np.angle(fft_result)
        
        # Create frequency axis using SciPy (or numpy fallback)
        frequencies = rfftfreq(fft_size, 1/sample_rate)
        
        # Compute spectral statistics
        total_power = np.sum(magnitude**2)
        spectral_centroid = np.sum(frequencies * magnitude**2) / total_power if total_power > 0 else 0
        
        # Frequency bands analysis with named bands
        frequency_bands = fft_utils.analyze_frequency_bands(frequencies, magnitude_db)
        
        # Find peaks
        peak_data = fft_utils.find_peaks(frequencies, magnitude_db)
        
        # Overall peak frequency and magnitude
        max_idx = np.argmax(magnitude_db)
        peak_frequency = float(frequencies[max_idx])
        peak_magnitude = float(magnitude_db[max_idx])
        
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
            'peaks': peak_data,
            'spectral_density': {
                'type': 'Magnitude Spectrum',
                'units': 'dB re FS',
                'description': 'Magnitude spectrum with proper window correction for frequency response analysis',
                'enbw_hz': float(enbw),
                'window_correction_applied': True
            }
        }
        
        # Add logarithmic summarization if requested (after normalization)
        if points_per_octave is not None:
            try:
                log_summary = fft_utils.summarize_fft_log_frequency(
                    frequencies, magnitude_db, points_per_octave
                )
                
                log_summary["smoothing"] = {"applied": False}
                
                # Apply normalization to log frequency summary if requested (after smoothing)
                if normalize is not None:
                    try:
                        log_freqs = np.array(log_summary['frequencies'])
                        log_mags = np.array(log_summary['magnitudes'])
                        
                        # Find closest frequency to normalization target
                        normalize_idx = np.argmin(np.abs(log_freqs - normalize))
                        normalize_freq_actual = log_freqs[normalize_idx]
                        normalize_level_db = log_mags[normalize_idx]
                        
                        # Normalize: subtract the reference level from all values
                        normalized_mags = log_mags - normalize_level_db
                        log_summary['magnitudes'] = normalized_mags.tolist()
                        
                        log_summary['normalization'] = {
                            "applied": True,
                            "requested_freq": float(normalize),
                            "actual_freq": float(normalize_freq_actual),
                            "reference_level_db": float(normalize_level_db)
                        }
                    except Exception as e:
                        log_summary['normalization'] = {
                            "applied": False,
                            "error": f"Normalization failed: {str(e)}"
                        }
                else:
                    log_summary['normalization'] = {"applied": False}
                
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
                              points_per_octave: int = None) -> Dict:
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

            # Normalize by window sum for proper amplitude scaling (coherent gain correction)
            mag = mag / window_sum
            power = mag * mag
            acc_power += power
            seg_count += 1

        if seg_count == 0:
            return compute_fft(audio_data, sample_rate, window_type, fft_size, normalize, 
                             points_per_octave)

        avg_power = acc_power / seg_count

        # Convert power to dB
        with np.errstate(divide='ignore', invalid='ignore'):
            magnitude_db = 10 * np.log10(avg_power + 1e-20)
            magnitude_db[~np.isfinite(magnitude_db)] = -np.inf

        smoothing_info = {"applied": False}

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
                log_summary = fft_utils.summarize_fft_log_frequency(
                    frequencies, magnitude_db, points_per_octave
                )
                
                log_summary["smoothing"] = {"applied": False}
                
                # Apply normalization to log frequency summary if requested (after smoothing)
                if normalize is not None:
                    try:
                        log_freqs = np.array(log_summary['frequencies'])
                        log_mags = np.array(log_summary['magnitudes'])
                        
                        # Find closest frequency to normalization target
                        normalize_idx = np.argmin(np.abs(log_freqs - normalize))
                        normalize_freq_actual = log_freqs[normalize_idx]
                        normalize_level_db = log_mags[normalize_idx]
                        
                        # Normalize: subtract the reference level from all values
                        normalized_mags = log_mags - normalize_level_db
                        log_summary['magnitudes'] = normalized_mags.tolist()
                        
                        log_summary['normalization'] = {
                            "applied": True,
                            "requested_freq": float(normalize),
                            "actual_freq": float(normalize_freq_actual),
                            "reference_level_db": float(normalize_level_db)
                        }
                    except Exception as e:
                        log_summary['normalization'] = {
                            "applied": False,
                            "error": f"Normalization failed: {str(e)}"
                        }
                else:
                    log_summary['normalization'] = {"applied": False}
                
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
                     points_per_octave: int = None,
                     use_time_averaging: bool = None) -> Dict:
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
        use_time_averaging: If True, use time-averaged FFT (better for sweeps). If None, auto-detect.
        
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
    
    # Auto-detect if time averaging should be used
    if use_time_averaging is None:
        # Use time averaging for longer signals that might be sweeps
        # and when we have enough data for multiple segments
        signal_duration = len(audio_data) / sample_rate
        suggested_fft_size = fft_size if fft_size is not None else min(32768, len(audio_data))
        potential_segments = len(audio_data) // (suggested_fft_size // 4)  # 75% overlap
        use_time_averaging = signal_duration > 2.0 and potential_segments >= 8
    
    # Perform FFT analysis
    if use_time_averaging:
        fft_result = compute_fft_time_averaged(audio_data, sample_rate, window_type, fft_size, 
                                             overlap=0.75, normalize=normalize, 
                                             points_per_octave=points_per_octave)
    else:
        fft_result = compute_fft(audio_data, sample_rate, window_type, fft_size, normalize, 
                               points_per_octave)
    
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


def get_fft_backend_info() -> Dict[str, Union[str, bool]]:
    """Get information about the FFT backend being used."""
    return {
        'backend': 'scipy' if HAS_SCIPY else 'numpy',
        'fft_module': 'scipy.fft' if HAS_SCIPY else 'numpy.fft',
        'window_module': 'scipy.signal.windows' if HAS_SCIPY else 'numpy (legacy)',
        'scipy_available': HAS_SCIPY,
        'description': 'High-performance SciPy FFT' if HAS_SCIPY else 'NumPy FFT fallback'
    }


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
