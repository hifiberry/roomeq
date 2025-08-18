#!/usr/bin/env python3
"""
Analyze the C reference implementation to understand proper logarithmic binning.
"""

import numpy as np
import sys
import os

# Add the src path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def analyze_c_reference_algorithm():
    """Analyze the key parts of the C reference implementation."""
    
    print("ANALYZING C REFERENCE IMPLEMENTATION")
    print("="*60)
    
    # Key parameters from the C code
    fminswp = 20.0
    fmaxswp = 20000.0
    numoptpts = 64  # Default number of output points
    sample_rate = 48000
    fftsize = 65536  # Example FFT size
    
    print(f"C Reference Parameters:")
    print(f"  fminswp: {fminswp} Hz")
    print(f"  fmaxswp: {fmaxswp} Hz") 
    print(f"  numoptpts: {numoptpts}")
    print(f"  sample_rate: {sample_rate} Hz")
    print(f"  fftsize: {fftsize}")
    
    # This is the key algorithm from reduce_samples() function:
    # double x = pow(fmaxswp / fminswp, 1.0/(double)numoptpts);
    # double vbw = (double)rate/((double)fftsize);
    # for(i = 0; i < numoptpts; i++){
    #     fr[i] = fminswp * pow(x, i);
    #     fend = fminswp * pow(x, i+1);
    #     ...
    #     j = (int) (fr[i]/vbw + 0.5);
    #     while (j * vbw < fend && j < fftsize / 2) {
    #         res[i] += resmag[j];
    #         pha[i] += respha[j];
    #         j++;
    #         cnt++;
    #     }
    #     res[i] /= cnt;  // ARITHMETIC MEAN!
    #     ...
    #     fr[i] = (fend - fr[i]) / log( fend / fr[i]);  // LOG AVERAGE OF FREQUENCIES
    # }
    
    print(f"\nC Algorithm Analysis:")
    
    # Calculate the frequency spacing factor
    x = pow(fmaxswp / fminswp, 1.0 / numoptpts)
    print(f"  Frequency spacing factor x: {x:.6f}")
    
    # Calculate video bandwidth (frequency resolution)
    vbw = sample_rate / fftsize
    print(f"  Video bandwidth (VBW): {vbw:.3f} Hz")
    
    # Calculate the logarithmic frequency points
    frequencies = []
    bin_ranges = []
    bin_counts = []
    
    print(f"\nC Logarithmic Binning (first 20 bins):")
    print(f"{'Bin':<4} {'Start(Hz)':<10} {'End(Hz)':<10} {'LogAvg(Hz)':<12} {'FFT Bins':<10} {'Method'}")
    print("-" * 70)
    
    for i in range(min(20, numoptpts)):
        # Start and end frequencies for this bin
        fr_start = fminswp * pow(x, i)
        fr_end = fminswp * pow(x, i + 1)
        
        # Calculate how many FFT bins are in this range
        start_bin = int(fr_start / vbw + 0.5)
        end_bin = start_bin
        
        # Count FFT bins that fall in this range
        fft_bin_count = 0
        j = start_bin
        while j * vbw < fr_end and j < fftsize // 2:
            fft_bin_count += 1
            j += 1
        
        # Calculate the logarithmic average frequency (this is the key!)
        if fr_end != fr_start:
            log_avg_freq = (fr_end - fr_start) / np.log(fr_end / fr_start)
        else:
            log_avg_freq = fr_start
        
        frequencies.append(log_avg_freq)
        bin_ranges.append((fr_start, fr_end))
        bin_counts.append(fft_bin_count)
        
        print(f"{i:<4} {fr_start:<10.1f} {fr_end:<10.1f} {log_avg_freq:<12.1f} {fft_bin_count:<10} {'MEAN'}")
    
    return frequencies, bin_ranges, bin_counts, x, vbw


def implement_c_algorithm():
    """Implement the C algorithm properly in Python."""
    
    print(f"\n" + "="*60)
    print("IMPLEMENTING C ALGORITHM IN PYTHON")
    print("="*60)
    
    # Test with our sine sweep data
    filename = "sox_signal.wav"
    
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found!")
        return
    
    from roomeq.fft import load_wav_file, compute_fft
    
    try:
        # Load audio data
        audio_data, sample_rate, metadata = load_wav_file(filename)
        
        # Compute raw FFT
        fft_result = compute_fft(
            audio_data=audio_data,
            sample_rate=sample_rate,
            window_type="hann",
            fft_size=None,
            normalize=None,
            points_per_octave=None  # Raw FFT only
        )
        
        frequencies = np.array(fft_result['frequencies'])
        magnitudes_db = np.array(fft_result['magnitudes'])
        fftsize = len(frequencies) * 2 - 1  # Original FFT size
        
        print(f"Input data:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  FFT size: {fftsize}")
        print(f"  Frequency bins: {len(frequencies)}")
        
        # Apply C algorithm
        fminswp = 20.0
        fmaxswp = 20000.0
        numoptpts = 64
        
        # Calculate spacing factor
        x = pow(fmaxswp / fminswp, 1.0 / numoptpts)
        vbw = sample_rate / fftsize
        
        print(f"C algorithm parameters:")
        print(f"  x (spacing factor): {x:.6f}")
        print(f"  VBW: {vbw:.3f} Hz")
        print(f"  Output points: {numoptpts}")
        
        # Process each logarithmic bin using C algorithm
        c_frequencies = []
        c_magnitudes = []
        
        for i in range(numoptpts):
            # Calculate bin range
            fr_start = fminswp * pow(x, i)
            fr_end = fminswp * pow(x, i + 1)
            
            # Find FFT bins in this range (exactly like C code)
            start_j = int(fr_start / vbw + 0.5)
            
            # Accumulate magnitudes from FFT bins in this range
            sum_magnitude = 0.0
            count = 0
            j = start_j
            
            while j * vbw < fr_end and j < len(frequencies):
                if j < len(magnitudes_db):
                    # Convert back from dB to linear for averaging
                    linear_magnitude = 10**(magnitudes_db[j] / 20.0)
                    sum_magnitude += linear_magnitude
                    count += 1
                j += 1
            
            if count > 0:
                # Average in linear space, then convert back to dB
                avg_linear = sum_magnitude / count
                avg_db = 20 * np.log10(avg_linear + 1e-20)
            else:
                avg_db = -100.0
            
            # Calculate logarithmic average frequency
            if fr_end != fr_start:
                log_avg_freq = (fr_end - fr_start) / np.log(fr_end / fr_start)
            else:
                log_avg_freq = fr_start
            
            c_frequencies.append(log_avg_freq)
            c_magnitudes.append(avg_db)
        
        # Calculate slope using C algorithm result
        c_frequencies = np.array(c_frequencies)
        c_magnitudes = np.array(c_magnitudes)
        
        # Use middle 80% for slope calculation
        start_idx = len(c_frequencies) // 10
        end_idx = len(c_frequencies) - len(c_frequencies) // 10
        
        freq_range = c_frequencies[start_idx:end_idx]
        mag_range = c_magnitudes[start_idx:end_idx]
        
        if len(freq_range) > 1:
            log_freq = np.log10(freq_range)
            c_slope, _ = np.polyfit(log_freq, mag_range, 1)
            
            print(f"\nC Algorithm Results:")
            print(f"  Slope: {c_slope:.2f} dB/decade")
            print(f"  Expected slope: 0.0 dB/decade")
            
            if abs(c_slope) < 3.0:
                print(f"  ✅ C algorithm produces good results!")
            else:
                print(f"  ❌ C algorithm still shows issues")
            
            return c_slope, c_frequencies, c_magnitudes
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def compare_with_our_implementation():
    """Compare C algorithm with our current implementation."""
    
    print(f"\n" + "="*60)
    print("COMPARING C ALGORITHM WITH OUR IMPLEMENTATION")
    print("="*60)
    
    # Get C algorithm result
    c_slope, c_freqs, c_mags = implement_c_algorithm()
    
    if c_slope is None:
        print("❌ C algorithm failed")
        return
    
    # Get our current algorithm result
    from roomeq.fft import analyze_wav_file
    
    try:
        our_result = analyze_wav_file(
            filepath="sox_signal.wav",
            window_type="hann", 
            fft_size=None,
            start_time=0,
            duration=None,
            normalize=None,
            points_per_octave=16
        )
        
        if our_result and 'fft_analysis' in our_result:
            fft_data = our_result['fft_analysis']
            
            if 'log_frequency_summary' in fft_data:
                log_data = fft_data['log_frequency_summary']
                our_frequencies = np.array(log_data['frequencies'])
                our_magnitudes = np.array(log_data['magnitudes'])
                
                # Calculate our slope
                start_idx = len(our_frequencies) // 10
                end_idx = len(our_frequencies) - len(our_frequencies) // 10
                
                freq_range = our_frequencies[start_idx:end_idx]
                mag_range = our_magnitudes[start_idx:end_idx]
                
                if len(freq_range) > 1:
                    log_freq = np.log10(freq_range)
                    our_slope, _ = np.polyfit(log_freq, mag_range, 1)
                    
                    print(f"COMPARISON RESULTS:")
                    print(f"{'Algorithm':<20} {'Slope (dB/dec)':<15} {'Points':<10} {'Status'}")
                    print(f"{'-'*60}")
                    print(f"{'C Reference':<20} {c_slope:<15.2f} {len(c_freqs):<10}")
                    
                    if abs(c_slope) < 3.0:
                        c_status = "GOOD"
                    else:
                        c_status = "POOR"
                    
                    print(f"{'Our Implementation':<20} {our_slope:<15.2f} {len(our_frequencies):<10}")
                    
                    if abs(our_slope) < 3.0:
                        our_status = "GOOD"
                    else:
                        our_status = "POOR"
                    
                    slope_diff = abs(our_slope - c_slope)
                    print(f"{'Difference':<20} {slope_diff:<15.2f}")
                    
                    if slope_diff < 2.0:
                        print(f"✅ Our implementation matches C reference closely!")
                    else:
                        print(f"❌ Significant difference from C reference")
                        print(f"⚠️  Need to adopt C algorithm approach")
                    
                    return c_slope, our_slope
    
    except Exception as e:
        print(f"❌ Error comparing implementations: {e}")
        return None, None


def main():
    """Main analysis function."""
    
    # Analyze the C reference
    analyze_c_reference_algorithm()
    
    # Compare implementations
    compare_with_our_implementation()


if __name__ == "__main__":
    main()
