"""
FFT Analysis Module with Rust Backend for RoomEQ Audio Processing

This module provides the same functionality as fft.py but uses the high-performance
Rust FFT analyzer backend for improved speed and memory efficiency. Falls back to
Python implementation if Rust backend is not available.
"""

import os
import subprocess
import tempfile
import csv
import numpy as np
from typing import Tuple, Dict, List, Union
from pathlib import Path

# Import the original Python implementation as fallback
from . import fft as fft_python
from . import fft_utils


class RustFFTAnalyzer:
    """
    High-performance FFT analyzer using the Rust backend.
    """
    
    def __init__(self, rust_binary_path: str = None):
        """
        Initialize the Rust FFT analyzer.
        
        Args:
            rust_binary_path: Path to the roomeq-analyzer binary. If None, searches standard locations.
        """
        self.rust_binary_path = rust_binary_path or self._find_rust_binary()
        self.available = self._check_availability()
        
    def _find_rust_binary(self) -> str:
        """Find the Rust FFT analyzer binary."""
        possible_paths = [
            # Development paths
            "/home/matuschd/hifiberry-os/packages/roomeq/roomeq/src/rust/target/release/roomeq-analyzer",
            "/home/matuschd/hifiberry-os/packages/roomeq/roomeq/src/rust/target/debug/roomeq-analyzer",
            # Relative paths
            "src/rust/target/release/roomeq-analyzer",
            "src/rust/target/debug/roomeq-analyzer",
            "../rust/target/release/roomeq-analyzer",
            "../rust/target/debug/roomeq-analyzer",
            # System path
            "roomeq-analyzer"
        ]
        
        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return os.path.abspath(path)
        
        # Try to find in PATH
        try:
            result = subprocess.run(['which', 'roomeq-analyzer'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return "roomeq-analyzer"  # Default fallback
    
    def _check_availability(self) -> bool:
        """Check if the Rust analyzer is available and working."""
        try:
            result = subprocess.run([self.rust_binary_path, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
    
    def analyze_file_pair(self, reference_path: str, measurement_path: str,
                         points_per_octave: int = 64, freq_min: float = 20.0,
                         freq_max: float = 20000.0, verbose: int = 0) -> Dict:
        """
        Analyze a measurement file against a reference using the Rust backend.
        
        Args:
            reference_path: Path to reference WAV file
            measurement_path: Path to measurement WAV file
            points_per_octave: Number of frequency points per octave
            freq_min: Minimum frequency for analysis
            freq_max: Maximum frequency for analysis
            verbose: Verbosity level (0-2)
            
        Returns:
            Dict containing FFT analysis results
            
        Raises:
            RuntimeError: If Rust analyzer fails or is not available
        """
        if not self.available:
            raise RuntimeError(f"Rust FFT analyzer not available at: {self.rust_binary_path}")
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            # Build command
            cmd = [
                self.rust_binary_path,
                '--reference', reference_path,
                '--points', str(points_per_octave),
                '--freq-min', str(freq_min),
                '--freq-max', str(freq_max),
                '--verbose', str(verbose),
                '--output', output_path,
                measurement_path
            ]
            
            # Run the Rust analyzer
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise RuntimeError(f"Rust analyzer failed: {result.stderr}")
            
            # Parse the CSV output
            frequencies = []
            magnitudes = []
            phases = []
            
            with open(output_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    frequencies.append(float(row['Frequency_Hz']))
                    magnitudes.append(float(row['Magnitude_dB']))
                    phases.append(float(row['Phase_degrees']))
            
            if not frequencies:
                raise RuntimeError("No data returned from Rust analyzer")
            
            # Get sample rate from the measurement file for metadata
            try:
                audio_data, sample_rate, metadata = fft_python.load_wav_file(measurement_path)
                file_metadata = metadata
            except Exception as e:
                # Fallback metadata if file loading fails
                sample_rate = 44100
                file_metadata = {
                    'sample_rate': 44100,
                    'duration': 0.0,
                    'channels': 1,
                    'bit_depth': 32,
                    'total_samples': 0,
                    'loaded_with': 'rust_analyzer'
                }
            
            # Calculate derived statistics
            max_idx = np.argmax(magnitudes)
            peak_frequency = frequencies[max_idx]
            peak_magnitude = magnitudes[max_idx]
            
            # Compute spectral centroid in linear domain
            linear_mags = [10**(m/10.0) for m in magnitudes]
            total_power = sum(linear_mags)
            spectral_centroid = sum(f * m for f, m in zip(frequencies, linear_mags)) / total_power if total_power > 0 else 0
            
            # Analyze frequency bands
            frequency_bands = fft_utils.analyze_frequency_bands(
                np.array(frequencies), np.array(magnitudes)
            )
            
            # Build result structure compatible with Python implementation
            result = {
                'fft_size': len(frequencies) * 2,  # Approximate for compatibility
                'window_type': 'hann',  # Rust analyzer uses Hann window
                'sample_rate': sample_rate,
                'frequency_resolution': (freq_max - freq_min) / len(frequencies),
                'frequencies': frequencies,
                'magnitudes': magnitudes,
                'phases': phases,
                'peak_frequency': float(peak_frequency),
                'peak_magnitude': float(peak_magnitude),
                'spectral_centroid': float(spectral_centroid),
                'frequency_bands': frequency_bands,
                'spectral_density': {
                    'type': 'Rust FFT Analysis',
                    'units': 'dB re Reference',
                    'description': 'High-performance FFT analysis with reference comparison',
                    'backend': 'rust',
                    'points_per_octave': points_per_octave,
                    'frequency_range': [freq_min, freq_max]
                },
                'log_frequency_summary': {
                    'frequencies': frequencies,
                    'magnitudes': magnitudes,
                    'points_per_octave': points_per_octave,
                    'frequency_range': [freq_min, freq_max],
                    'n_points': len(frequencies),
                    'algorithm': 'rust_implementation',
                    'normalization': {'applied': True, 'reference_file': reference_path},
                    'smoothing': {'applied': False}
                }
            }
            
            return result
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(output_path)
            except OSError:
                pass


# Global analyzer instance
_rust_analyzer = None


def get_rust_analyzer() -> RustFFTAnalyzer:
    """Get the global Rust analyzer instance."""
    global _rust_analyzer
    if _rust_analyzer is None:
        _rust_analyzer = RustFFTAnalyzer()
    return _rust_analyzer


def load_wav_file(filepath: str) -> Tuple[np.ndarray, int, Dict]:
    """
    Load WAV file and return audio data, sample rate, and metadata.
    
    This function maintains compatibility with the Python implementation.
    
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
    return fft_python.load_wav_file(filepath)


def compute_fft(audio_data: np.ndarray, sample_rate: int, window_type: str = 'hann', 
                fft_size: int = None, normalize: float = None, points_per_octave: int = None,
                psychoacoustic_smoothing: float = None) -> Dict:
    """
    Compute comprehensive FFT analysis of audio data.
    
    This function uses the Rust backend when possible for improved performance,
    especially for reference comparisons and logarithmic frequency analysis.
    
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
    # For single-file analysis without reference, use Python implementation
    # Rust backend is optimized for reference comparisons
    return fft_python.compute_fft(
        audio_data, sample_rate, window_type, fft_size, normalize,
        points_per_octave, psychoacoustic_smoothing
    )


def compute_fft_time_averaged(audio_data: np.ndarray, sample_rate: int, window_type: str = 'hann',
                              fft_size: int = None, overlap: float = 0.5, normalize: float = None,
                              points_per_octave: int = None, psychoacoustic_smoothing: float = None) -> Dict:
    """
    Compute a time-averaged FFT over multiple overlapping windows (Welch-style averaging).
    
    Uses Python implementation as Rust backend is optimized for single-window analysis.
    
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
    return fft_python.compute_fft_time_averaged(
        audio_data, sample_rate, window_type, fft_size, overlap, normalize,
        points_per_octave, psychoacoustic_smoothing
    )


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
    return fft_python.analyze_wav_file(
        filepath, window_type, fft_size, start_time, duration, normalize,
        points_per_octave, psychoacoustic_smoothing
    )


def analyze_wav_file_with_reference(measurement_path: str, reference_path: str,
                                   points_per_octave: int = 64, 
                                   freq_min: float = 20.0, freq_max: float = 20000.0,
                                   window_type: str = 'hann', verbose: int = 0,
                                   use_rust_backend: bool = True) -> Dict:
    """
    Analyze a WAV file against a reference using the optimal backend.
    
    This function is optimized for reference comparisons and will use the Rust
    backend when available for better performance, especially with logarithmic
    frequency analysis and multi-file comparisons.
    
    Args:
        measurement_path: Path to measurement WAV file
        reference_path: Path to reference WAV file
        points_per_octave: Number of frequency points per octave
        freq_min: Minimum frequency for analysis
        freq_max: Maximum frequency for analysis
        window_type: Window function to apply
        verbose: Verbosity level (0-2)
        use_rust_backend: Whether to prefer Rust backend when available
        
    Returns:
        Dict containing comprehensive analysis results
        
    Raises:
        RuntimeError: If analysis fails
    """
    analyzer = get_rust_analyzer()
    
    # Try Rust backend first if available and requested
    if use_rust_backend and analyzer.available:
        try:
            rust_result = analyzer.analyze_file_pair(
                reference_path, measurement_path, points_per_octave,
                freq_min, freq_max, verbose
            )
            
            # Load file metadata for compatibility
            try:
                audio_data, sample_rate, metadata = load_wav_file(measurement_path)
                file_info = {
                    'filename': os.path.basename(measurement_path),
                    'original_metadata': metadata,
                    'analyzed_duration': metadata.get('duration', 0.0),
                    'analyzed_samples': metadata.get('total_samples', 0),
                    'start_time': 0.0,
                    'backend': 'rust'
                }
            except Exception:
                # Fallback file info
                file_info = {
                    'filename': os.path.basename(measurement_path),
                    'backend': 'rust',
                    'error': 'Could not load file metadata'
                }
            
            return {
                'file_info': file_info,
                'fft_analysis': rust_result,
                'reference_file': reference_path,
                'backend_used': 'rust'
            }
            
        except Exception as e:
            if verbose > 0:
                print(f"Rust backend failed, falling back to Python: {e}")
    
    # Fallback to Python implementation
    # This requires implementing reference comparison in Python
    # For now, analyze both files and compute the difference
    try:
        # Analyze measurement file
        measurement_result = analyze_wav_file(
            measurement_path, window_type=window_type,
            points_per_octave=points_per_octave, psychoacoustic_smoothing=None
        )
        
        # Analyze reference file
        reference_result = analyze_wav_file(
            reference_path, window_type=window_type,
            points_per_octave=points_per_octave, psychoacoustic_smoothing=None
        )
        
        # Extract log frequency data if available
        measurement_fft = measurement_result['fft_analysis']
        reference_fft = reference_result['fft_analysis']
        
        if ('log_frequency_summary' in measurement_fft and 
            'log_frequency_summary' in reference_fft):
            
            meas_log = measurement_fft['log_frequency_summary']
            ref_log = reference_fft['log_frequency_summary']
            
            # Compute difference (measurement - reference)
            meas_freqs = np.array(meas_log['frequencies'])
            meas_mags = np.array(meas_log['magnitudes'])
            ref_freqs = np.array(ref_log['frequencies'])
            ref_mags = np.array(ref_log['magnitudes'])
            
            # Interpolate reference to measurement frequency points
            ref_interp = np.interp(meas_freqs, ref_freqs, ref_mags)
            diff_mags = meas_mags - ref_interp
            
            # Create result structure
            result_fft = measurement_fft.copy()
            result_fft['frequencies'] = meas_freqs.tolist()
            result_fft['magnitudes'] = diff_mags.tolist()
            result_fft['log_frequency_summary']['magnitudes'] = diff_mags.tolist()
            result_fft['log_frequency_summary']['normalization'] = {
                'applied': True,
                'reference_file': reference_path,
                'method': 'difference'
            }
            result_fft['spectral_density']['description'] = 'Measurement vs Reference (Python backend)'
            result_fft['spectral_density']['backend'] = 'python'
            
        else:
            # Use full resolution difference if log summary not available
            meas_freqs = np.array(measurement_fft['frequencies'])
            meas_mags = np.array(measurement_fft['magnitudes'])
            ref_freqs = np.array(reference_fft['frequencies'])
            ref_mags = np.array(reference_fft['magnitudes'])
            
            # Interpolate reference to measurement frequency points
            ref_interp = np.interp(meas_freqs, ref_freqs, ref_mags)
            diff_mags = meas_mags - ref_interp
            
            result_fft = measurement_fft.copy()
            result_fft['magnitudes'] = diff_mags.tolist()
        
        return {
            'file_info': measurement_result['file_info'],
            'fft_analysis': result_fft,
            'reference_file': reference_path,
            'backend_used': 'python'
        }
        
    except Exception as e:
        raise RuntimeError(f"Analysis failed with both backends: {e}")


def get_supported_window_types() -> List[str]:
    """Get list of supported window function types."""
    return fft_python.get_supported_window_types()


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
    return fft_python.validate_fft_parameters(fft_size, window_type)


def is_rust_backend_available() -> bool:
    """Check if the Rust FFT backend is available."""
    return get_rust_analyzer().available


def get_backend_info() -> Dict:
    """Get information about available FFT backends."""
    analyzer = get_rust_analyzer()
    return {
        'python': {
            'available': True,
            'description': 'Pure Python implementation with NumPy FFT',
            'features': ['single_file', 'time_averaging', 'windowing', 'smoothing']
        },
        'rust': {
            'available': analyzer.available,
            'binary_path': analyzer.rust_binary_path if analyzer.available else None,
            'description': 'High-performance Rust implementation',
            'features': ['reference_comparison', 'multi_file', 'logarithmic_binning', 'fast_processing']
        },
        'default_backend': 'rust' if analyzer.available else 'python'
    }
