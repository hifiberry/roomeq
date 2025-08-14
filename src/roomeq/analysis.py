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
            self.device, self.microphone_sensitivity = self._detect_microphone()
        else:
            self.device = device
            self.microphone_sensitivity = None
    
    def _detect_microphone(self):
        detector = MicrophoneDetector()
        microphones = detector.detect_microphones()
        
        if not microphones:
            raise RuntimeError("No microphone detected")
            
        mic = microphones[0]
        card_id, name, sensitivity_str = mic
        device = f"hw:{card_id},0"
        
        try:
            sensitivity = float(sensitivity_str)
        except (ValueError, TypeError):
            sensitivity = None
            
        return device, sensitivity
    
    def record_audio(self, duration_seconds=1.0):
        pcm = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            alsaaudio.PCM_NORMAL,
            device=self.device,
            channels=self.channels,
            rate=self.sample_rate,
            format=self.format_type,
            periodsize=1024
        )
        
        total_samples = int(self.sample_rate * duration_seconds)
        samples_collected = 0
        audio_data = []
        
        while samples_collected < total_samples:
            length, data = pcm.read()
            if length > 0:
                samples = np.frombuffer(data, dtype=np.int16)
                audio_data.extend(samples.tolist())
                samples_collected += len(samples)
        
        pcm.close()
        return np.array(audio_data, dtype=np.int16)
    
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
        return self.microphone_sensitivity + rms_db_fs
    
    def analyze_recording(self, duration_seconds=1.0):
        samples = self.record_audio(duration_seconds)
        rms_db_fs = self.calculate_rms_db(samples)
        rms_db_spl = self.calculate_rms_spl(rms_db_fs)
        
        return {
            'rms_db_fs': rms_db_fs,
            'rms_db_spl': rms_db_spl,
            'samples_count': len(samples),
            'device': self.device
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
        result = analyzer.analyze_recording(args.time)
        
        print(f"Relative RMS: {result['rms_db_fs']:.2f} dB FS")
        if result['rms_db_spl'] is not None:
            print(f"Absolute RMS: {result['rms_db_spl']:.2f} dB SPL")
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
