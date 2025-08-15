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
                fft_size: int = None, normalize: float = None) -> Dict:
    """
    Compute comprehensive FFT analysis of audio data.
    
    Args:
        audio_data: Normalized audio data [-1, 1]
        sample_rate: Sample rate in Hz
        window_type: Window function ('hann', 'hamming', 'blackman', 'none')
        fft_size: FFT size (power of 2), auto-calculated if None
        normalize: Frequency in Hz to normalize to 0 dB, None for no normalization
        
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
        
        # Compute magnitude spectrum (in dB)
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)  # Add small value to avoid log(0)
        
        # Compute phase spectrum
        phase = np.angle(fft_result)
        
        # Create frequency axis
        frequencies = np.fft.rfftfreq(fft_size, 1/sample_rate)
        
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
        
        return {
            'fft_size': fft_size,
            'window_type': window_type,
            'sample_rate': sample_rate,
            'frequency_resolution': float(sample_rate / fft_size),
            'frequencies': frequencies.tolist(),
            'magnitudes': magnitude_db.tolist(),
            'phases': phase.tolist(),
            'peak_frequency': peak_frequency,
            'peak_magnitude': peak_magnitude,
            'spectral_centroid': float(spectral_centroid),
            'frequency_bands': frequency_bands,
            'normalization': normalization_info
        }
        
    except Exception as e:
        raise RuntimeError(f"FFT computation failed: {str(e)}")


def analyze_wav_file(filepath: str, window_type: str = 'hann', fft_size: int = None,
                     start_time: float = 0.0, duration: float = None, normalize: float = None) -> Dict:
    """
    Complete FFT analysis of a WAV file with time windowing support.
    
    Args:
        filepath: Path to WAV file
        window_type: Window function to apply
        fft_size: FFT size (power of 2), auto-calculated if None
        start_time: Start analysis at this time in seconds
        duration: Duration to analyze in seconds, None for entire file
        normalize: Frequency in Hz to normalize to 0 dB, None for no normalization
        
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
    fft_result = compute_fft(audio_data, sample_rate, window_type, fft_size, normalize)
    
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
