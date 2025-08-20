#!/usr/bin/env python3
"""
Create Frequency Response Plots for WAV Files

This script processes all WAV files in a given directory and creates frequency response plots
for each file. The plots are saved as PNG files with the same base name as the WAV files.

Usage:
    python3 create_frequency_response.py <directory> [options]

Examples:
    python3 create_frequency_response.py /path/to/wav/files
    python3 create_frequency_response.py . --show
    python3 create_frequency_response.py recordings --window blackman --freq-range 50 15000
"""

import sys
import os
import argparse
import glob
from pathlib import Path
from typing import List, Tuple

# Add the src directory to Python path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from roomeq.fft import plot_wav_frequency_response


def find_wav_files(directory: str) -> List[str]:
    """
    Find all WAV files in the given directory.
    
    Args:
        directory: Directory to search for WAV files
        
    Returns:
        List of WAV file paths
    """
    wav_files = []
    
    # Search for .wav files (case insensitive)
    patterns = ['*.wav', '*.WAV']
    for pattern in patterns:
        wav_files.extend(glob.glob(os.path.join(directory, pattern)))
    
    # Sort for consistent ordering
    wav_files.sort()
    
    return wav_files


def process_wav_files(directory: str, 
                     show_plots: bool = False,
                     window_type: str = 'hann',
                     fft_size: int = None,
                     freq_range: Tuple[float, float] = (20, 20000),
                     mag_range: Tuple[float, float] = None,
                     output_dir: str = None) -> None:
    """
    Process all WAV files in directory and create frequency response plots.
    
    Args:
        directory: Directory containing WAV files
        show_plots: Whether to display plots interactively
        window_type: FFT window type
        fft_size: FFT size (None for auto)
        freq_range: Frequency range for plots
        mag_range: Magnitude range for plots (None for auto)
        output_dir: Output directory for plots (None to save alongside WAV files)
    """
    # Find WAV files
    wav_files = find_wav_files(directory)
    
    if not wav_files:
        print(f"No WAV files found in directory: {directory}")
        return
    
    print(f"Found {len(wav_files)} WAV file(s) in {directory}")
    print(f"Window: {window_type}, FFT size: {fft_size or 'auto'}")
    print(f"Frequency range: {freq_range[0]}-{freq_range[1]} Hz")
    print()
    
    # Process each WAV file
    successful = 0
    failed = 0
    
    for i, wav_path in enumerate(wav_files, 1):
        try:
            # Determine output path
            wav_filename = os.path.splitext(os.path.basename(wav_path))[0]
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{wav_filename}.png")
            else:
                output_path = None  # Will save alongside WAV file
            
            print(f"[{i}/{len(wav_files)}] Processing: {os.path.basename(wav_path)}")
            
            # Create frequency response plot
            plot_path = plot_wav_frequency_response(
                wav_path=wav_path,
                output_path=output_path,
                show_plot=show_plots,
                window_type=window_type,
                fft_size=fft_size,
                freq_range=freq_range,
                mag_range=mag_range
            )
            
            print(f"    → Saved: {os.path.basename(plot_path)}")
            successful += 1
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            failed += 1
    
    print()
    print(f"Processing complete: {successful} successful, {failed} failed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create frequency response plots for WAV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/recordings/
  %(prog)s . --show
  %(prog)s recordings --window blackman --freq-range 50 15000
  %(prog)s samples --output-dir plots --mag-range -60 10
        """
    )
    
    parser.add_argument('directory', 
                       help='Directory containing WAV files to process')
    
    parser.add_argument('--show', action='store_true',
                       help='Display plots interactively (default: False)')
    
    parser.add_argument('--window', default='hann',
                       choices=['hann', 'hamming', 'blackman', 'bartlett', 'flat'],
                       help='FFT window type (default: hann)')
    
    parser.add_argument('--fft-size', type=int,
                       help='FFT size (default: auto-detected)')
    
    parser.add_argument('--freq-range', nargs=2, type=float, 
                       default=[20, 20000], metavar=('MIN', 'MAX'),
                       help='Frequency range for plots in Hz (default: 20 20000)')
    
    parser.add_argument('--mag-range', nargs=2, type=float,
                       metavar=('MIN', 'MAX'),
                       help='Magnitude range for plots in dB (default: auto)')
    
    parser.add_argument('--output-dir', 
                       help='Directory to save plots (default: alongside WAV files)')
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: Directory does not exist: {args.directory}")
        sys.exit(1)
    
    # Convert arguments
    freq_range = tuple(args.freq_range)
    mag_range = tuple(args.mag_range) if args.mag_range else None
    
    # Validate frequency range
    if freq_range[0] >= freq_range[1]:
        print("Error: Minimum frequency must be less than maximum frequency")
        sys.exit(1)
    
    # Validate magnitude range
    if mag_range and mag_range[0] >= mag_range[1]:
        print("Error: Minimum magnitude must be less than maximum magnitude")
        sys.exit(1)
    
    try:
        # Process WAV files
        process_wav_files(
            directory=args.directory,
            show_plots=args.show,
            window_type=args.window,
            fft_size=args.fft_size,
            freq_range=freq_range,
            mag_range=mag_range,
            output_dir=args.output_dir
        )
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
