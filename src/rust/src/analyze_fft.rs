//! FFT Analysis Module - Rust Port
//!
//! This is a Rust port of the C FFT implementation from HiFiBerry measurements.
//! Original C code by Joerg Schambacher, 2019, i2Audio GmbH
//! 
//! Provides FFT analysis functionality for WAV files with frequency response
//! comparison and multi-file averaging capabilities.

use std::f64::consts::PI;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Noise floor constant for dB calculations
const NOISE_FLOOR: f64 = 1e-15;

/// Convert linear amplitude to dB
#[inline]
fn to_db(x: f64) -> f64 {
    20.0 * (x + NOISE_FLOOR).log10()
}

/// Convert dB to linear amplitude
#[inline]
fn from_db(x: f64) -> f64 {
    10.0_f64.powf(x / 20.0)
}

/// Square function
#[inline]
fn sqr(x: f64) -> f64 {
    x * x
}

/// WAV file header structure
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct WavHeader {
    pub riff: [u8; 4],
    pub filesize: u32,
    pub wave: [u8; 4],
    pub fmt: [u8; 4],
    pub fmtlen: u32,
    pub fmttype: u16,
    pub channels: u16,
    pub samplerate: u32,
    pub bytespersec: u32,
    pub framesize: u16,
    pub bitspersample: u16,
    pub data: [u8; 4],
    pub datasize: u32,
}

/// FFT Analysis configuration
#[derive(Debug, Clone)]
pub struct FftConfig {
    pub verbose: u32,
    pub num_output_points: usize,
    pub f_min_sweep: f64,
    pub f_max_sweep: f64,
}

impl Default for FftConfig {
    fn default() -> Self {
        Self {
            verbose: 0,
            num_output_points: 64,
            f_min_sweep: 20.0,
            f_max_sweep: 20000.0,
        }
    }
}

/// FFT Analysis results
#[derive(Debug, Clone)]
pub struct FftResult {
    pub frequencies: Vec<f64>,
    pub magnitudes: Vec<f64>,
    pub phases: Vec<f64>,
    pub sample_rate: u32,
    pub fft_size: usize,
}

/// Multi-file FFT analyzer
pub struct FftAnalyzer {
    config: FftConfig,
    reference_mag: Option<Vec<f64>>,
    reference_pha: Option<Vec<f64>>,
    accumulated_mag: Vec<f64>,
    accumulated_pha: Vec<f64>,
    file_count: usize,
    fft_size: usize,
    sample_rate: u32,
}

impl FftAnalyzer {
    /// Create a new FFT analyzer with default configuration
    pub fn new() -> Self {
        Self::with_config(FftConfig::default())
    }

    /// Create a new FFT analyzer with custom configuration
    pub fn with_config(config: FftConfig) -> Self {
        Self {
            config,
            reference_mag: None,
            reference_pha: None,
            accumulated_mag: Vec::new(),
            accumulated_pha: Vec::new(),
            file_count: 0,
            fft_size: 0,
            sample_rate: 0,
        }
    }

    /// Set the reference file for comparison
    pub fn set_reference<P: AsRef<Path>>(&mut self, reference_path: P) -> Result<(), Box<dyn std::error::Error>> {
        let (pcm_data, header) = self.read_wav_file(reference_path)?;
        
        // Initialize analyzer parameters from reference file
        self.fft_size = pcm_data.len();
        self.sample_rate = header.samplerate;

        if self.config.verbose > 0 {
            println!("Reference file: {} samples at {} Hz", self.fft_size, self.sample_rate);
        }

        // Convert to f64 and apply Hann window
        let mut real: Vec<f64> = Vec::with_capacity(self.fft_size);
        let mut window_sum = 0.0;
        
        for i in 0..self.fft_size {
            let pcm_val = if i < pcm_data.len() { pcm_data[i] as f64 } else { 0.0 };
            // Apply Hann window
            let window_val = 0.5 * (1.0 - (2.0 * PI * i as f64 / (self.fft_size - 1) as f64).cos());
            let windowed_val = pcm_val * window_val;
            real.push(windowed_val);
            window_sum += window_val;
        }
        let mut imag = vec![0.0; self.fft_size];

        self.fft_mono(&mut real, &mut imag)?;

        // Calculate magnitude and phase, normalize with window correction
        let mut ref_mag = Vec::with_capacity(self.fft_size / 2);
        let mut ref_pha = Vec::with_capacity(self.fft_size / 2);
        let mut max_mag = 0.0;

        // Apply single-sided spectrum scaling and proper normalization
        for i in 0..self.fft_size / 2 {
            let mut magnitude = (sqr(real[i]) + sqr(imag[i])).sqrt();
            
            // Scale for single-sided spectrum (double all except DC)
            if i > 0 {
                magnitude *= 2.0;
            }
            
            // Apply window correction (coherent gain)
            magnitude /= window_sum;
            
            let phase = if sqr(real[i]) > NOISE_FLOOR {
                imag[i].atan2(real[i])
            } else {
                0.0
            };

            let mag_db = to_db(magnitude);
            ref_mag.push(mag_db);
            ref_pha.push(phase);

            if mag_db > max_mag {
                max_mag = mag_db;
            }
        }

        // Normalize reference to 0dB peak
        for mag in &mut ref_mag {
            *mag -= max_mag;
        }

        self.reference_mag = Some(ref_mag);
        self.reference_pha = Some(ref_pha);

        // Initialize accumulation arrays
        self.accumulated_mag = vec![0.0; self.fft_size / 2];
        self.accumulated_pha = vec![0.0; self.fft_size / 2];

        Ok(())
    }

    /// Process a measurement file
    pub fn process_file<P: AsRef<Path>>(&mut self, file_path: P) -> Result<(), Box<dyn std::error::Error>> {
        if self.reference_mag.is_none() {
            return Err("Reference file must be set before processing measurement files".into());
        }

        let (pcm_data, header) = self.read_wav_file(file_path.as_ref())?;

        if header.samplerate != self.sample_rate {
            let sample_rate_val = header.samplerate;
            return Err(format!(
                "Sample rate mismatch: expected {}, got {}", 
                self.sample_rate, sample_rate_val
            ).into());
        }

        if self.config.verbose > 0 {
            println!("Processing: {} ({} samples)", 
                file_path.as_ref().display(), pcm_data.len());
        }

        // Pad or truncate to match reference size and apply Hann window
        let mut real: Vec<f64> = Vec::with_capacity(self.fft_size);
        let mut window_sum = 0.0;
        
        for i in 0..self.fft_size {
            let pcm_val = if i < pcm_data.len() { pcm_data[i] as f64 } else { 0.0 };
            // Apply Hann window
            let window_val = 0.5 * (1.0 - (2.0 * PI * i as f64 / (self.fft_size - 1) as f64).cos());
            let windowed_val = pcm_val * window_val;
            real.push(windowed_val);
            window_sum += window_val;
        }
        let mut imag = vec![0.0; self.fft_size];

        self.fft_mono(&mut real, &mut imag)?;

        // Calculate magnitude and phase for this file
        let mut pcm_mag = Vec::with_capacity(self.fft_size / 2);
        let mut pcm_pha = Vec::with_capacity(self.fft_size / 2);

        for i in 0..self.fft_size / 2 {
            let mut magnitude = (sqr(real[i]) + sqr(imag[i])).sqrt();
            
            // Scale for single-sided spectrum (double all except DC)
            if i > 0 {
                magnitude *= 2.0;
            }
            
            // Apply window correction (coherent gain)
            magnitude /= window_sum;
            
            let phase = if sqr(real[i]) > NOISE_FLOOR {
                imag[i].atan2(real[i])
            } else {
                0.0
            };

            pcm_mag.push(magnitude);
            pcm_pha.push(phase);
        }

        // Accumulate results (running average)
        if self.file_count == 0 {
            // First file
            for i in 0..self.fft_size / 2 {
                self.accumulated_mag[i] = pcm_mag[i];
                self.accumulated_pha[i] = pcm_pha[i];
            }
        } else {
            // Average with previous results
            for i in 0..self.fft_size / 2 {
                self.accumulated_mag[i] = (self.accumulated_mag[i] + pcm_mag[i]) / 2.0;
                self.accumulated_pha[i] = (self.accumulated_pha[i] + pcm_pha[i]) / 2.0;
            }
        }

        self.file_count += 1;
        Ok(())
    }

    /// Get the final analysis result with frequency reduction
    pub fn get_result(&self) -> Result<FftResult, Box<dyn std::error::Error>> {
        if self.reference_mag.is_none() || self.file_count == 0 {
            return Err("No reference or measurement data available".into());
        }

        let ref_mag = self.reference_mag.as_ref().unwrap();
        let ref_pha = self.reference_pha.as_ref().unwrap();

        // Calculate response relative to reference
        let mut response_mag = Vec::with_capacity(self.fft_size / 2);
        let mut response_pha = Vec::with_capacity(self.fft_size / 2);

        for i in 0..self.fft_size / 2 {
            let mag_db = to_db(self.accumulated_mag[i]);
            response_mag.push(mag_db - ref_mag[i]);
            response_pha.push(self.accumulated_pha[i] - ref_pha[i]);
        }

        // Reduce to logarithmic frequency bins
        self.reduce_samples(&response_mag, &response_pha)
    }

    /// Reduce samples to logarithmic frequency distribution
    fn reduce_samples(&self, mag_data: &[f64], pha_data: &[f64]) -> Result<FftResult, Box<dyn std::error::Error>> {
        let num_points = self.config.num_output_points;
        let mut frequencies = Vec::with_capacity(num_points);
        let mut magnitudes = Vec::with_capacity(num_points);
        let mut phases = Vec::with_capacity(num_points);

        let x = (self.config.f_max_sweep / self.config.f_min_sweep).powf(1.0 / num_points as f64);
        let vbw = self.sample_rate as f64 / self.fft_size as f64; // Video bandwidth

        if self.config.verbose > 0 {
            println!("FFT size: {}, VBW: {:.1} Hz", self.fft_size, vbw);
        }

        for i in 0..num_points {
            let f_start = self.config.f_min_sweep * x.powi(i as i32);
            let f_end = self.config.f_min_sweep * x.powi(i as i32 + 1);

            if self.config.verbose > 1 {
                println!("Averaging from {:.2} to {:.2} Hz", f_start, f_end);
            }

            let mut mag_sum = 0.0;
            let mut pha_sum = 0.0;
            let mut count = 0;

            let start_bin = (f_start / vbw + 0.5) as usize;
            let mut bin = start_bin;

            while bin as f64 * vbw < f_end && bin < self.fft_size / 2 {
                if bin < mag_data.len() {
                    mag_sum += mag_data[bin];
                    pha_sum += pha_data[bin];
                    count += 1;
                }
                bin += 1;
            }

            if count > 0 {
                magnitudes.push(mag_sum / count as f64);
                phases.push(pha_sum / count as f64 * 180.0 / PI);
                
                // Calculate logarithmic center frequency
                let log_center = (f_end - f_start) / (f_end / f_start).ln();
                frequencies.push(log_center);
            } else {
                magnitudes.push(0.0);
                phases.push(0.0);
                frequencies.push((f_start + f_end) / 2.0);
            }
        }

        Ok(FftResult {
            frequencies,
            magnitudes,
            phases,
            sample_rate: self.sample_rate,
            fft_size: self.fft_size,
        })
    }

    /// In-place FFT computation (Cooley-Tukey algorithm)
    fn fft_mono(&self, real: &mut [f64], imag: &mut [f64]) -> Result<(), Box<dyn std::error::Error>> {
        let n = real.len();
        if n != imag.len() || !n.is_power_of_two() {
            return Err("FFT input arrays must have equal power-of-2 length".into());
        }

        let m = (n as f64).log2() as u32;

        // Bit reversal
        let mut j = 0;
        for i in 0..n - 1 {
            if i < j {
                real.swap(i, j);
                imag.swap(i, j);
            }
            
            let mut k = n >> 1;
            while k <= j {
                j -= k;
                k >>= 1;
            }
            j += k;
        }

        // FFT computation
        let mut c1 = -1.0;
        let mut c2 = 0.0;
        let mut l2 = 1;

        for _ in 0..m {
            let l1 = l2;
            l2 <<= 1;
            let mut u1 = 1.0;
            let mut u2 = 0.0;

            for j in 0..l1 {
                let mut i = j;
                while i < n {
                    let i1 = i + l1;
                    let t1 = u1 * real[i1] - u2 * imag[i1];
                    let t2 = u1 * imag[i1] + u2 * real[i1];
                    
                    real[i1] = real[i] - t1;
                    imag[i1] = imag[i] - t2;
                    real[i] += t1;
                    imag[i] += t2;
                    
                    i += l2;
                }
                
                let z = u1 * c1 - u2 * c2;
                u2 = u1 * c2 + u2 * c1;
                u1 = z;
            }
            
            c2 = ((1.0 - c1) / 2.0).sqrt();
            c2 = -c2;
            c1 = ((1.0 + c1) / 2.0).sqrt();
        }

        // Normalize by FFT size (not by N/2)
        let norm_factor = n as f64;
        for i in 0..n {
            real[i] /= norm_factor;
            imag[i] /= norm_factor;
        }

        Ok(())
    }

    /// Read WAV file and return PCM data
    fn read_wav_file<P: AsRef<Path>>(&self, path: P) -> Result<(Vec<i32>, WavHeader), Box<dyn std::error::Error>> {
        let mut file = File::open(&path)?;
        
        // Read WAV header
        let mut header_bytes = [0u8; std::mem::size_of::<WavHeader>()];
        file.read_exact(&mut header_bytes)?;
        
        let header: WavHeader = unsafe { std::mem::transmute(header_bytes) };

        // Validate header
        if &header.riff != b"RIFF" || &header.wave != b"WAVE" {
            return Err("Invalid WAV file format".into());
        }

        if header.channels != 1 || header.bitspersample != 32 {
            return Err("Only 32-bit mono WAV files are supported".into());
        }

        if self.config.verbose > 1 {
            let channels = header.channels;
            let bits = header.bitspersample;
            let rate = header.samplerate;
            let data_size = header.datasize;
            println!("WAV info: {} channels, {} bit @ {} Hz", 
                channels, bits, rate);
            println!("Data size: {} bytes", data_size);
        }

        // Calculate number of samples
        let data_len = (header.filesize + 8 - 44) as usize;
        let num_samples = data_len / 4; // 32-bit = 4 bytes per sample

        // Find next power of 2 for FFT
        let fft_size = next_power_of_2(num_samples);
        
        // Read PCM data
        let mut pcm_data = vec![0i32; fft_size];
        let mut temp_buffer = vec![0u8; data_len];
        file.read_exact(&mut temp_buffer)?;

        // Convert bytes to i32 samples (little-endian)
        for (i, chunk) in temp_buffer.chunks_exact(4).enumerate() {
            if i < num_samples {
                pcm_data[i] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }

        if self.config.verbose > 1 {
            println!("Read {} samples, padded to {} for FFT", num_samples, fft_size);
        }

        Ok((pcm_data, header))
    }

    /// Get the FFT size that would be used for a WAV file
    pub fn get_fft_size<P: AsRef<Path>>(path: P) -> Result<usize, Box<dyn std::error::Error>> {
        let mut file = File::open(&path)?;
        
        let mut header_bytes = [0u8; std::mem::size_of::<WavHeader>()];
        file.read_exact(&mut header_bytes)?;
        
        let header: WavHeader = unsafe { std::mem::transmute(header_bytes) };

        if &header.riff != b"RIFF" || &header.wave != b"WAVE" {
            return Err("Invalid WAV file format".into());
        }

        if header.channels != 1 || header.bitspersample != 32 {
            return Err("Only 32-bit mono WAV files are supported".into());
        }

        let data_len = (header.filesize + 8 - 44) as usize;
        let num_samples = data_len / 4;
        
        Ok(next_power_of_2(num_samples))
    }
}

/// Find the next power of 2 greater than or equal to n
fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    
    let mut power = 1;
    while power < n {
        power <<= 1;
    }
    power
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(0), 1);
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(2), 2);
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(8), 8);
        assert_eq!(next_power_of_2(9), 16);
        assert_eq!(next_power_of_2(1000), 1024);
    }

    #[test]
    fn test_db_conversion() {
        assert!((to_db(1.0) - 0.0).abs() < 1e-10);
        assert!((to_db(10.0) - 20.0).abs() < 1e-10);
        assert!((from_db(0.0) - 1.0).abs() < 1e-10);
        assert!((from_db(20.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_fft_config_default() {
        let config = FftConfig::default();
        assert_eq!(config.verbose, 0);
        assert_eq!(config.num_output_points, 64);
        assert_eq!(config.f_min_sweep, 20.0);
        assert_eq!(config.f_max_sweep, 20000.0);
    }
}
