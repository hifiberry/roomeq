#!/usr/bin/env python3
"""
Signal generator command-line tool.

Simple interface for generating noise and sine sweeps.
"""

import sys
import os
import argparse
import signal

# Add the src path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'roomeq'))
from signal_generator import SignalGenerator


def main():
    parser = argparse.ArgumentParser(description='Audio Signal Generator Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Noise subcommand
    noise_parser = subparsers.add_parser('noise', help='Play white noise')
    noise_parser.add_argument('-d', '--device', type=str, default=None,
                            help='ALSA output device (e.g., hw:0,0)')
    noise_parser.add_argument('-t', '--time', type=float, default=0,
                            help='Duration in seconds (0 = infinite)')
    noise_parser.add_argument('-a', '--amplitude', type=float, default=0.5,
                            help='Amplitude (0.0-1.0, default: 0.5)')
    
    # Sweep subcommand
    sweep_parser = subparsers.add_parser('sweep', help='Play sine sweep')
    sweep_parser.add_argument('-d', '--device', type=str, default=None,
                            help='ALSA output device (e.g., hw:0,0)')
    sweep_parser.add_argument('-s', '--start', type=float, default=20,
                            help='Start frequency Hz (default: 20)')
    sweep_parser.add_argument('-e', '--end', type=float, default=20000,
                            help='End frequency Hz (default: 20000)')
    sweep_parser.add_argument('-t', '--time', type=float, default=10,
                            help='Duration in seconds (default: 10)')
    sweep_parser.add_argument('-a', '--amplitude', type=float, default=0.5,
                            help='Amplitude (0.0-1.0, default: 0.5)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Create generator
    generator = SignalGenerator(device=args.device)
    
    # Setup signal handler for clean shutdown
    def handle_interrupt(sig, frame):
        print("\nStopping...")
        generator.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_interrupt)
    
    try:
        if args.command == 'noise':
            generator.play_noise(duration=args.time, amplitude=args.amplitude)
            generator.wait_for_completion()
            
        elif args.command == 'sweep':
            generator.play_sine_sweep(
                start_freq=args.start,
                end_freq=args.end, 
                duration=args.time,
                amplitude=args.amplitude
            )
            generator.wait_for_completion()
            
    except KeyboardInterrupt:
        generator.stop()
        
    return 0


if __name__ == "__main__":
    exit(main())
