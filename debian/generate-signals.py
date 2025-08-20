#!/usr/bin/env python3
"""
Generate pre-created signal files for the roomeq package.
This script creates high-quality test signals during package build.
"""

import numpy as np
import wave
import os
import sys
import argparse
from pathlib import Path

def generate_logarithmic_sweep(output_path, start_freq=10.0, end_freq=22000.0, 
                              duration=10.0, sample_rate=48000, amplitude=0.8):
    """Generate a logarithmic frequency sweep signal."""
    print(f"Generating logarithmic sweep: {start_freq} Hz → {end_freq} Hz, {duration}s")
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate logarithmic sweep using manual calculation for better control
    # f(t) = f0 * (f1/f0)^(t/T) where f0=start_freq, f1=end_freq, T=duration
    instantaneous_freq = start_freq * np.power(end_freq / start_freq, t / duration)
    
    # Generate the signal by integrating the instantaneous frequency
    # Phase = 2π * ∫f(t)dt = 2π * f0 * T * (f1/f0)^(t/T) / ln(f1/f0)
    k = np.log(end_freq / start_freq)
    phase = 2 * np.pi * start_freq * duration * (np.power(end_freq / start_freq, t / duration) - 1) / k
    y = np.sin(phase)
    
    # Apply fade-in/fade-out to avoid clicks (10ms each)
    fade_samples = int(0.01 * sample_rate)
    if fade_samples > 0:
        y[:fade_samples] *= np.linspace(0, 1, fade_samples)
        y[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # Scale to desired amplitude and convert to 16-bit
    y_scaled = (y * amplitude * 32767).astype(np.int16)
    
    # Create stereo signal (identical left and right channels)
    stereo_signal = np.column_stack((y_scaled, y_scaled))
    
    # Write WAV file
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(2)  # Stereo
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(stereo_signal.tobytes())
    
    file_size = output_path.stat().st_size
    print(f"Generated: {output_path.name} ({file_size:,} bytes)")
    return output_path

def generate_white_noise(output_path, duration=5.0, sample_rate=48000, amplitude=0.5):
    """Generate a white noise signal."""
    print(f"Generating white noise: {duration}s")
    
    # Generate white noise
    samples = int(sample_rate * duration)
    y = np.random.normal(0, 1, samples)
    
    # Apply fade-in/fade-out to avoid clicks (10ms each)
    fade_samples = int(0.01 * sample_rate)
    if fade_samples > 0:
        y[:fade_samples] *= np.linspace(0, 1, fade_samples)
        y[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # Scale to desired amplitude and convert to 16-bit
    y_scaled = (y * amplitude * 32767).astype(np.int16)
    
    # Create stereo signal
    stereo_signal = np.column_stack((y_scaled, y_scaled))
    
    # Write WAV file
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(2)  # Stereo
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(stereo_signal.tobytes())
    
    file_size = output_path.stat().st_size
    print(f"Generated: {output_path.name} ({file_size:,} bytes)")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate signal files for roomeq package')
    parser.add_argument('output_dir', help='Output directory for signal files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating signal files in: {output_dir}")
    
    try:
        # Check if required dependencies are available
        try:
            import scipy.signal
            use_scipy = True
            if args.verbose:
                print("Using scipy for signal generation")
        except ImportError:
            use_scipy = False
            if args.verbose:
                print("scipy not available, using numpy-only implementation")
        
        # Generate logarithmic sweep (10Hz to 22kHz, 10 seconds)
        sweep_file = output_dir / "sweep_10hz_22000hz_10s.wav"
        generate_logarithmic_sweep(sweep_file)
        
        # Generate white noise (5 seconds) - optional for future use
        # noise_file = output_dir / "noise_white_5s.wav" 
        # generate_white_noise(noise_file)
        
        print(f"\nSignal generation completed successfully!")
        print(f"Files generated in: {output_dir}")
        
        # List generated files
        wav_files = list(output_dir.glob("*.wav"))
        total_size = sum(f.stat().st_size for f in wav_files)
        print(f"Generated {len(wav_files)} signal file(s), total size: {total_size:,} bytes")
        
        return 0
        
    except ImportError as e:
        print(f"Error: Missing required Python packages: {e}", file=sys.stderr)
        print("Please install: python3-numpy python3-scipy", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error generating signals: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
