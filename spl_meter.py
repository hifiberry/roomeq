#!/usr/bin/env python3
"""
Simple SPL measurement command-line tool.

Measures SPL level using the first detected microphone or a specified device.
"""

import sys
import os
import argparse

# Add the src path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'roomeq'))
from analysis import measure_spl


def main():
    parser = argparse.ArgumentParser(description='SPL Level Measurement Tool')
    parser.add_argument('-d', '--device', type=str, default=None,
                       help='ALSA device name (e.g., hw:1,0). If not specified, auto-detects first microphone.')
    parser.add_argument('-t', '--time', type=float, default=1.0,
                       help='Measurement duration in seconds (default: 1.0)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed information')
    
    args = parser.parse_args()
    
    print(f"Measuring SPL for {args.time} seconds...")
    
    # Perform measurement
    result = measure_spl(device=args.device, duration=args.time)
    
    if not result['success']:
        print(f"ERROR: {result['error']}")
        return 1
    
    # Display results
    if result['spl_db'] is not None:
        print(f"SPL Level: {result['spl_db']:.1f} dB SPL")
    else:
        print("SPL Level: N/A (no microphone sensitivity data)")
    
    if args.verbose:
        print(f"\nDetailed Results:")
        print(f"  Device: {result['device']}")
        print(f"  Relative level: {result['rms_db_fs']:.2f} dB FS")
        print(f"  Measurement duration: {result['duration']} seconds")
        if result['microphone_sensitivity']:
            print(f"  Microphone sensitivity: {result['microphone_sensitivity']} dB SPL")
        if result['microphone_gain'] is not None:
            print(f"  Microphone gain: {result['microphone_gain']} dB")
            effective_sens = result['microphone_sensitivity'] - result['microphone_gain']
            print(f"  Effective sensitivity: {effective_sens} dB SPL")
    
    return 0


if __name__ == "__main__":
    exit(main())
