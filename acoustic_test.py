#!/usr/bin/env python3
"""
Acoustic measurement test script.

Demonstrates the complete workflow:
1. Generate test signal (noise or sweep)
2. Measure SPL levels
3. Provide results

This script can be used for system calibration and testing.
"""

import sys
import os
import time
import threading
import argparse
import signal

# Add the src path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'roomeq'))
from signal_generator import SignalGenerator
from analysis import measure_spl
from microphone import MicrophoneDetector


class AcousticMeasurement:
    def __init__(self, output_device=None, input_device=None):
        self.generator = SignalGenerator(device=output_device)
        self.input_device = input_device
        self.detector = MicrophoneDetector()
        self.running = False
        
    def measure_noise_floor(self, duration=2.0):
        """Measure the noise floor (ambient noise) without signal generation."""
        print(f"Measuring noise floor for {duration} seconds...")
        return measure_spl(duration=duration, device=self.input_device)
        
    def measure_with_signal(self, signal_type="noise", duration=5.0, amplitude=0.5, **kwargs):
        """Generate signal and measure SPL simultaneously."""
        print(f"Starting {signal_type} generation and SPL measurement...")
        
        # Start signal generation in background
        if signal_type == "noise":
            self.generator.play_noise(duration=duration + 1, amplitude=amplitude)  # Extra second for measurement
        elif signal_type == "sweep":
            start_freq = kwargs.get('start_freq', 20)
            end_freq = kwargs.get('end_freq', 20000)
            self.generator.play_sine_sweep(
                start_freq=start_freq,
                end_freq=end_freq,
                duration=duration + 1,
                amplitude=amplitude
            )
        
        # Wait a moment for signal to stabilize
        time.sleep(0.5)
        
        # Measure SPL
        spl_result = measure_spl(duration=duration, device=self.input_device)
        
        # Stop generation
        self.generator.stop()
        
        return spl_result
        
    def calibration_test(self):
        """Run a complete calibration test sequence."""
        print("=== Acoustic Measurement Calibration Test ===")
        print()
        
        # Show available devices
        microphones = self.detector.detect_microphones()
        print("Available input devices:")
        for card_idx, name, sensitivity, gain in microphones:
            print(f"  {card_idx}: {name} (sensitivity: {sensitivity}dB, gain: {gain}dB)")
        print()
        
        # Measure noise floor
        try:
            result = self.measure_noise_floor(duration=2.0)
            noise_floor = result['spl_db']
            print(f"Noise floor: {noise_floor:.1f} dB SPL")
            print()
        except Exception as e:
            print(f"Error measuring noise floor: {e}")
            return
        
        # Test different signal levels
        test_levels = [0.1, 0.3, 0.5, 0.7]
        print("Testing different signal amplitudes:")
        
        for amplitude in test_levels:
            try:
                print(f"  Testing amplitude {amplitude} ({amplitude*100:.0f}%)...")
                result = self.measure_with_signal("noise", duration=3.0, amplitude=amplitude)
                spl = result['spl_db']
                snr = spl - noise_floor
                print(f"    SPL: {spl:.1f} dB, SNR: {snr:.1f} dB")
            except Exception as e:
                print(f"    Error: {e}")
        
        print()
        print("Test completed!")
        
    def stop(self):
        """Stop all operations."""
        self.generator.stop()


def main():
    parser = argparse.ArgumentParser(description='Acoustic Measurement Test Tool')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output device (e.g., hw:0,0)')
    parser.add_argument('-i', '--input', type=int, default=None,
                       help='Input device index (default: auto-detect)')
    parser.add_argument('-t', '--test', choices=['calibration', 'noise', 'sweep'],
                       default='calibration', help='Test type to run')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Test duration in seconds')
    parser.add_argument('--amplitude', type=float, default=0.5,
                       help='Signal amplitude (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Create measurement system
    measurement = AcousticMeasurement(
        output_device=args.output,
        input_device=args.input
    )
    
    # Setup signal handler
    def handle_interrupt(sig, frame):
        print("\nStopping...")
        measurement.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_interrupt)
    
    try:
        if args.test == 'calibration':
            measurement.calibration_test()
        elif args.test == 'noise':
            result = measurement.measure_noise_floor(2.0)
            noise_floor = result['spl_db']
            result = measurement.measure_with_signal("noise", args.duration, args.amplitude)
            spl = result['spl_db']
            print(f"Noise floor: {noise_floor:.1f} dB SPL")
            print(f"Signal level: {spl:.1f} dB SPL")
            print(f"SNR: {spl - noise_floor:.1f} dB")
        elif args.test == 'sweep':
            result = measurement.measure_noise_floor(2.0)
            noise_floor = result['spl_db']
            result = measurement.measure_with_signal("sweep", args.duration, args.amplitude)
            spl = result['spl_db']
            print(f"Noise floor: {noise_floor:.1f} dB SPL")
            print(f"Sweep level: {spl:.1f} dB SPL")
            print(f"SNR: {spl - noise_floor:.1f} dB")
            
    except KeyboardInterrupt:
        measurement.stop()
    except Exception as e:
        print(f"Error: {e}")
        measurement.stop()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
