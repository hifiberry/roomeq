#!/usr/bin/env python3
"""
Audio analysis module for RoomEQ.
"""

import time
import logging
from typing import Optional, Tuple
import numpy as np
import alsaaudio

try:
    from .microphone import MicrophoneDetector
except ImportError:
    from microphone import MicrophoneDetector

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Audio analyzer for ALSA recordings."""
    
    def __init__(self, device=None, sample_rate=48000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.format_type = alsaaudio.PCM_FORMAT_S16_LE
        
        if device is None:
            self.device, self.microphone_sensitivity, self.microphone_gain = self._detect_microphone()
        else:
            self.device = device
            # Try to get sensitivity and gain for the specified device
            self.microphone_sensitivity, self.microphone_gain = self._get_device_properties(device)
    
    def _detect_microphone(self):
        detector = MicrophoneDetector()
        microphones = detector.detect_microphones()
        
        if not microphones:
            raise RuntimeError("No microphone detected")
            
        mic = microphones[0]
        card_id, name, sensitivity_str, gain_db = mic
        device = f"hw:{card_id},0"
        
        try:
            sensitivity = float(sensitivity_str)
        except (ValueError, TypeError):
            sensitivity = None
            
        return device, sensitivity, gain_db
    
    def _get_device_properties(self, device):
        """Get sensitivity and gain for a specific device."""
        try:
            detector = MicrophoneDetector()
            microphones = detector.detect_microphones()
            
            for mic in microphones:
                card_id, name, sensitivity_str, gain_db = mic
                mic_device = f"hw:{card_id},0"
                
                if mic_device == device:
                    try:
                        sensitivity = float(sensitivity_str)
                    except (ValueError, TypeError):
                        sensitivity = None
                    return sensitivity, gain_db
                    
            return None, None
            
        except Exception as e:
            return None, None
    
    def record_audio(self, duration_seconds=1.0):
        """
        Record audio using arecord as a workaround for alsaaudio PY_SSIZE_T_CLEAN issue.
        This is more reliable than the alsaaudio PCM.read() method.
        """
        import subprocess
        import tempfile
        import os
        
        # Create a temporary file for the recording
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Calculate exact number of samples needed
            total_samples = int(self.sample_rate * duration_seconds)
            
            # Use arecord to capture audio with exact sample count
            cmd = [
                'arecord',
                '-D', self.device,
                '-f', 'S16_LE',
                '-c', str(self.channels),
                '-r', str(self.sample_rate),
                '-s', str(total_samples),  # Use sample count instead of duration
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_seconds + 5)
            
            if result.returncode != 0:
                raise RuntimeError(f"arecord failed: {result.stderr}")
            
            # Read the recorded WAV file
            import wave
            with wave.open(temp_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                audio_data = np.frombuffer(frames, dtype=np.int16)
            
            return audio_data
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Recording timeout after {duration_seconds + 5} seconds")
        except Exception as e:
            raise RuntimeError(f"Failed to record audio: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def calculate_rms_db(self, samples):
        if len(samples) == 0:
            return -float('inf')
            
        normalized = samples.astype(np.float64) / 32768.0
        rms_linear = np.sqrt(np.mean(normalized ** 2))
        
        if rms_linear > 0:
            return 20.0 * np.log10(rms_linear)
        else:
            return -float('inf')
    
    def calculate_rms_spl(self, rms_db_fs):
        if self.microphone_sensitivity is None:
            return None
            
        # Start with base sensitivity (max SPL at 0 dBFS)
        effective_sensitivity = self.microphone_sensitivity
        
        # If gain is applied, it reduces the effective maximum SPL
        # e.g., with 20 dB gain, 0 dBFS now represents (sensitivity - 20) dB SPL
        if self.microphone_gain is not None:
            effective_sensitivity = self.microphone_sensitivity - self.microphone_gain
            
        return effective_sensitivity + rms_db_fs
    
    def analyze_recording(self, duration_seconds=1.0):
        samples = self.record_audio(duration_seconds)
        rms_db_fs = self.calculate_rms_db(samples)
        rms_db_spl = self.calculate_rms_spl(rms_db_fs)
        
        return {
            'rms_db_fs': rms_db_fs,
            'rms_db_spl': rms_db_spl,
            'samples_count': len(samples),
            'device': self.device,
            'microphone_sensitivity': self.microphone_sensitivity,
            'microphone_gain': self.microphone_gain
        }

def measure_spl(device=None, duration=1.0):
    """
    Simple function to measure SPL level.
    
    Args:
        device: ALSA device name (e.g., "hw:1,0"). If None, auto-detects first microphone.
        duration: Measurement duration in seconds
        
    Returns:
        Dictionary with SPL measurement results
    """
    try:
        analyzer = AudioAnalyzer(device=device)
        result = analyzer.analyze_recording(duration)
        
        return {
            'spl_db': result['rms_db_spl'],
            'rms_db_fs': result['rms_db_fs'],
            'device': result['device'],
            'duration': duration,
            'microphone_sensitivity': result['microphone_sensitivity'],
            'microphone_gain': result['microphone_gain'],
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'spl_db': None,
            'rms_db_fs': None,
            'device': device,
            'duration': duration,
            'success': False,
            'error': str(e)
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Audio RMS analyzer')
    parser.add_argument('-t', '--time', type=float, default=1.0, help='Duration')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')
    args = parser.parse_args()
    
    try:
        analyzer = AudioAnalyzer()
        print(f"Recording {args.time}s from {analyzer.device}...")
        if analyzer.microphone_sensitivity and analyzer.microphone_gain is not None:
            effective_sens = analyzer.microphone_sensitivity - analyzer.microphone_gain
            print(f"Microphone: sensitivity={analyzer.microphone_sensitivity} dB SPL, gain={analyzer.microphone_gain} dB, effective={effective_sens} dB SPL")
        elif analyzer.microphone_sensitivity:
            print(f"Microphone: sensitivity={analyzer.microphone_sensitivity} dB SPL, gain=N/A")
            
        result = analyzer.analyze_recording(args.time)
        
        print(f"Relative RMS: {result['rms_db_fs']:.2f} dB FS")
        if result['rms_db_spl'] is not None:
            print(f"Absolute RMS: {result['rms_db_spl']:.2f} dB SPL")
        
        if args.verbose:
            print(f"Gain-adjusted calculation: {result['microphone_sensitivity']:.1f} - {result['microphone_gain']:.1f} + {result['rms_db_fs']:.2f} = {result['rms_db_spl']:.2f} dB SPL")
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
