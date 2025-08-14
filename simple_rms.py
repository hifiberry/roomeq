#!/usr/bin/env python3
"""
Simple RMS level measurement tool.

Detects microphone, records 1 second at S16_LE 48kHz, 
and displays relative and absolute RMS levels.
"""

import sys
import os
import time
import numpy as np
import alsaaudio

# Import microphone detection directly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'roomeq'))
from microphone import MicrophoneDetector


def detect_microphone():
    """Detect the first available microphone."""
    try:
        detector = MicrophoneDetector()
        microphones = detector.detect_microphones()
        
        if not microphones:
            print("ERROR: No microphone detected")
            sys.exit(1)
            
        # Get first microphone
        mic = microphones[0]
        card_id, name, sensitivity_str = mic
        device = f"hw:{card_id},0"
        
        try:
            sensitivity = float(sensitivity_str)
        except (ValueError, TypeError):
            print(f"WARNING: Invalid sensitivity for {name}, using None")
            sensitivity = None
            
        print(f"Using microphone: {name} ({device})")
        if sensitivity:
            print(f"Sensitivity: {sensitivity} dB SPL")
            
        return device, sensitivity
        
    except Exception as e:
        print(f"ERROR: Failed to detect microphone: {e}")
        sys.exit(1)


def record_and_measure(device, sensitivity):
    """Record 1 second and measure RMS levels."""
    
    # Audio settings - exactly as specified
    sample_rate = 48000
    channels = 1
    duration = 1.0
    format_type = alsaaudio.PCM_FORMAT_S16_LE
    
    print(f"\nRecording 1 second (S16_LE, 48kHz, mono)...")
    
    try:
        # Open PCM device
        pcm = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            alsaaudio.PCM_NORMAL,
            device=device,
            channels=channels,
            rate=sample_rate,
            format=format_type,
            periodsize=1024
        )
        
        # Calculate total samples needed
        total_samples = int(sample_rate * duration)
        samples_collected = 0
        audio_data = []
        
        start_time = time.time()
        timeout = duration + 2.0
        
        # Record audio
        while samples_collected < total_samples:
            if time.time() - start_time > timeout:
                print("ERROR: Recording timeout")
                sys.exit(1)
                
            length, data = pcm.read()
            if length > 0:
                samples = np.frombuffer(data, dtype=np.int16)
                audio_data.extend(samples.tolist())
                samples_collected += len(samples)
        
        pcm.close()
        
        # Convert to numpy array
        audio_array = np.array(audio_data, dtype=np.int16)
        print(f"Recorded {len(audio_array)} samples")
        
        # Calculate RMS
        # Normalize to [-1, 1] range
        normalized = audio_array.astype(np.float64) / 32768.0
        
        # RMS calculation
        rms_linear = np.sqrt(np.mean(normalized ** 2))
        
        if rms_linear > 0:
            rms_db_fs = 20.0 * np.log10(rms_linear)
        else:
            rms_db_fs = -float('inf')
        
        # Display results
        print(f"\n=== RMS Results ===")
        print(f"Relative RMS: {rms_db_fs:.2f} dB FS")
        
        if sensitivity is not None:
            rms_db_spl = sensitivity + rms_db_fs
            print(f"Absolute RMS: {rms_db_spl:.2f} dB SPL")
        else:
            print("Absolute RMS: N/A (no sensitivity data)")
            
    except alsaaudio.ALSAAudioError as e:
        print(f"ERROR: ALSA error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def main():
    print("=== Simple RMS Level Measurement ===")
    
    # Detect microphone
    device, sensitivity = detect_microphone()
    
    # Record and measure
    record_and_measure(device, sensitivity)
    
    print("\nMeasurement complete.")


if __name__ == "__main__":
    main()
